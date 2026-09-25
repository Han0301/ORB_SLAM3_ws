#include <chrono>
#include <cmath>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "MapPoint.h"
#include "System.h"

class OrbSlam3RgbdNode final : public rclcpp::Node {
 public:
  OrbSlam3RgbdNode() : Node("orbslam3_rgbd") {
    const auto vocabulary = declare_parameter<std::string>("vocabulary", "");
    const auto settings = declare_parameter<std::string>("settings", "");
    color_topic_ = declare_parameter<std::string>("color_topic", "/camera/color/image_raw");
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
    map_frame_ = declare_parameter<std::string>("map_frame", "orb_map");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "orb_camera");
    max_path_poses_ = declare_parameter<int>("max_path_poses", 10000);
    use_imu_ = declare_parameter<bool>("use_imu", false);
    const bool enable_viewer = declare_parameter<bool>("enable_viewer", false);
    imu_topic_ = declare_parameter<std::string>("imu_topic", "/camera/gyro_accel/sample");

    if (vocabulary.empty() || settings.empty()) {
      throw std::runtime_error("parameters 'vocabulary' and 'settings' must be set");
    }

    RCLCPP_INFO(get_logger(), "Loading ORB vocabulary: %s", vocabulary.c_str());
    const auto sensor = use_imu_ ? ORB_SLAM3::System::IMU_RGBD : ORB_SLAM3::System::RGBD;
    slam_ = std::make_unique<ORB_SLAM3::System>(vocabulary, settings, sensor, enable_viewer);
    RCLCPP_INFO(get_logger(), "ORB-SLAM3 Pangolin viewer: %s", enable_viewer ? "enabled" : "disabled");

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("pose", 10);
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 10);
    points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("map_points", 2);
    state_pub_ = create_publisher<std_msgs::msg::Int32>("tracking_state", 10);
    imu_initialized_pub_ = create_publisher<std_msgs::msg::Bool>("imu_initialized", 10);
    imu_samples_pub_ = create_publisher<std_msgs::msg::UInt32>("imu_samples_per_frame", 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    color_sub_.subscribe(this, color_topic_, rmw_qos_profile_sensor_data);
    depth_sub_.subscribe(this, depth_topic_, rmw_qos_profile_sensor_data);
    sync_ = std::make_shared<Synchronizer>(SyncPolicy(20), color_sub_, depth_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.035));
    sync_->registerCallback(
        std::bind(&OrbSlam3RgbdNode::imageCallback, this,
                  std::placeholders::_1, std::placeholders::_2));
    if (use_imu_) {
      imu_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
      rclcpp::SubscriptionOptions imu_options;
      imu_options.callback_group = imu_callback_group_;
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
          imu_topic_, rclcpp::SensorDataQoS(),
          std::bind(&OrbSlam3RgbdNode::imuCallback, this, std::placeholders::_1), imu_options);
    }
    path_.header.frame_id = map_frame_;
    RCLCPP_INFO(get_logger(), "Waiting for RGB-D%s: %s + %s",
                use_imu_ ? " + IMU" : "", color_topic_.c_str(), depth_topic_.c_str());
    if (use_imu_) RCLCPP_INFO(get_logger(), "IMU topic: %s", imu_topic_.c_str());
  }

  ~OrbSlam3RgbdNode() override {
    if (slam_) slam_->Shutdown();
  }

 private:
  using Image = sensor_msgs::msg::Image;
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
  using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    const double stamp = rclcpp::Time(msg->header.stamp).seconds();
    if (!imu_queue_.empty() && stamp <= imu_queue_.back().t) {
      ++imu_non_monotonic_;
      return;
    }
    imu_queue_.emplace_back(
        static_cast<float>(msg->linear_acceleration.x),
        static_cast<float>(msg->linear_acceleration.y),
        static_cast<float>(msg->linear_acceleration.z),
        static_cast<float>(msg->angular_velocity.x),
        static_cast<float>(msg->angular_velocity.y),
        static_cast<float>(msg->angular_velocity.z), stamp);
    while (imu_queue_.size() > 4000) imu_queue_.pop_front();
  }

  std::vector<ORB_SLAM3::IMU::Point> takeImuUntil(double image_stamp) {
    std::vector<ORB_SLAM3::IMU::Point> samples;
    if (!use_imu_) return samples;
    std::lock_guard<std::mutex> lock(imu_mutex_);
    while (!imu_queue_.empty() && imu_queue_.front().t <= image_stamp) {
      if (last_image_stamp_ < 0.0 || imu_queue_.front().t > last_image_stamp_) {
        samples.push_back(imu_queue_.front());
      }
      imu_queue_.pop_front();
    }
    return samples;
  }

  static cv::Mat depthMeters(const Image::ConstSharedPtr &msg) {
    if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
        msg->encoding == sensor_msgs::image_encodings::MONO16) {
      cv::Mat out;
      cv_bridge::toCvShare(msg, msg->encoding)->image.convertTo(out, CV_32F, 0.001);
      return out;
    }
    if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      return cv_bridge::toCvCopy(msg, msg->encoding)->image;
    }
    throw std::runtime_error("unsupported depth encoding: " + msg->encoding);
  }

  void imageCallback(const Image::ConstSharedPtr &color_msg,
                     const Image::ConstSharedPtr &depth_msg) {
    std::lock_guard<std::mutex> lock(track_mutex_);
    try {
      const cv::Mat color = cv_bridge::toCvCopy(
          color_msg, sensor_msgs::image_encodings::BGR8)->image;
      const cv::Mat depth = depthMeters(depth_msg);
      const double stamp = rclcpp::Time(color_msg->header.stamp).seconds();
      const auto imu_samples = takeImuUntil(stamp);
      if (use_imu_ && imu_samples.size() < 2) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Waiting for at least two IMU samples before processing RGB-D");
        return;
      }
      const Sophus::SE3f tcw = slam_->TrackRGBD(color, depth, stamp, imu_samples);
      last_image_stamp_ = stamp;

      std_msgs::msg::UInt32 imu_count;
      imu_count.data = static_cast<uint32_t>(imu_samples.size());
      imu_samples_pub_->publish(imu_count);

      std_msgs::msg::Int32 state;
      state.data = slam_->GetTrackingState();
      state_pub_->publish(state);
      std_msgs::msg::Bool imu_initialized;
      imu_initialized.data = use_imu_ && slam_->IsImuInitialized();
      imu_initialized_pub_->publish(imu_initialized);
      if (state.data != ORB_SLAM3::Tracking::OK) return;

      const Sophus::SE3f twc = tcw.inverse();
      publishPose(twc, color_msg->header.stamp);
      publishMapPoints(color_msg->header.stamp);
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "cv_bridge: %s", e.what());
    } catch (const std::exception &e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "RGB-D tracking: %s", e.what());
    }
  }

  geometry_msgs::msg::Pose poseFrom(const Sophus::SE3f &twc) const {
    geometry_msgs::msg::Pose pose;
    const Eigen::Vector3f t = twc.translation();
    const Eigen::Quaternionf q(twc.rotationMatrix());
    pose.position.x = t.x();
    pose.position.y = t.y();
    pose.position.z = t.z();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();
    return pose;
  }

  void publishPose(const Sophus::SE3f &twc, const builtin_interfaces::msg::Time &stamp) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = stamp;
    pose.header.frame_id = map_frame_;
    pose.pose = poseFrom(twc);
    pose_pub_->publish(pose);

    nav_msgs::msg::Odometry odom;
    odom.header = pose.header;
    odom.child_frame_id = camera_frame_;
    odom.pose.pose = pose.pose;
    odom_pub_->publish(odom);

    path_.header.stamp = stamp;
    path_.poses.push_back(pose);
    if (max_path_poses_ > 0 && static_cast<int>(path_.poses.size()) > max_path_poses_) {
      path_.poses.erase(path_.poses.begin(),
                        path_.poses.begin() + (path_.poses.size() - max_path_poses_));
    }
    path_pub_->publish(path_);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = pose.header;
    tf.child_frame_id = camera_frame_;
    tf.transform.translation.x = pose.pose.position.x;
    tf.transform.translation.y = pose.pose.position.y;
    tf.transform.translation.z = pose.pose.position.z;
    tf.transform.rotation = pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);
  }

  void publishMapPoints(const builtin_interfaces::msg::Time &stamp) {
    const auto map_points = slam_->GetAllMapPoints();
    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header.stamp = stamp;
    cloud.header.frame_id = map_frame_;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    std::size_t valid = 0;
    for (auto *point : map_points) {
      if (point != nullptr && !point->isBad()) ++valid;
    }
    modifier.resize(valid);
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> z(cloud, "z");
    for (auto *point : map_points) {
      if (point == nullptr || point->isBad()) continue;
      const Eigen::Vector3f p = point->GetWorldPos();
      *x = p.x(); *y = p.y(); *z = p.z();
      ++x; ++y; ++z;
    }
    points_pub_->publish(cloud);
  }

  std::unique_ptr<ORB_SLAM3::System> slam_;
  message_filters::Subscriber<Image> color_sub_;
  message_filters::Subscriber<Image> depth_sub_;
  std::shared_ptr<Synchronizer> sync_;
  std::mutex track_mutex_;
  std::mutex imu_mutex_;
  std::deque<ORB_SLAM3::IMU::Point> imu_queue_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  std::string color_topic_, depth_topic_, imu_topic_, map_frame_, camera_frame_;
  bool use_imu_{false};
  double last_image_stamp_{-1.0};
  uint64_t imu_non_monotonic_{0};
  int max_path_poses_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr imu_initialized_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr imu_samples_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  nav_msgs::msg::Path path_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<OrbSlam3RgbdNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception &e) {
    std::cerr << "ORB-SLAM3 ROS 2 fatal error: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
