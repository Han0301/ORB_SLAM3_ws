#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

#include <unistd.h>

#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "MapPoint.h"
#include "rgbd_node.h"

OrbSlam3RgbdNode::OrbSlam3RgbdNode() : Node("orbslam3_rgbd")
{
  RCLCPP_INFO(get_logger(),
              "----------------------------------------------------------------Initializing OrbSlam3RgbdNode");
  // 1 declare the parameters for the node
  // change the value in the yaml and override the default values in the launch file
  const auto vocabulary = declare_parameter<std::string>("vocabulary", "");
  const auto settings = declare_parameter<std::string>("settings", "");

  if (vocabulary.empty() || settings.empty())
  {
    throw std::runtime_error("parameters 'vocabulary' and 'settings' must be set");
  }

  cv::FileStorage configuration(settings, cv::FileStorage::READ);
  if (!configuration.isOpened())
  {
    throw std::runtime_error("cannot open ORB-SLAM3 settings: " + settings);
  }

  color_topic_ = declare_parameter<std::string>(
    "color_topic", readString(configuration, "ROS.color_topic", "/camera/color/image_raw"));
  depth_topic_ = declare_parameter<std::string>(
    "depth_topic", readString(configuration, "ROS.depth_topic", "/camera/depth/image_raw"));
  imu_topic_ = declare_parameter<std::string>(
    "imu_topic", readString(configuration, "ROS.imu_topic", "/camera/gyro_accel/sample"));

  use_imu_ = declare_parameter<bool>(
    "use_imu", readBool(configuration, "ROS.use_imu", true));

  map_frame_ = declare_parameter<std::string>(
    "map_frame", readString(configuration, "ROS.map_frame", "orb_map"));
  camera_frame_ = declare_parameter<std::string>(
    "camera_frame", readString(configuration, "ROS.camera_frame", "orb_camera"));
  max_path_poses_ = declare_parameter<int>(
    "max_path_poses", readInt(configuration, "ROS.max_path_poses", 10000));
  output_stable_frames_required_ = declare_parameter<int>(
    "output_stable_frames",
    readInt(configuration, "ROS.output_stable_frames", 30));
  pose_rebase_translation_threshold_ = declare_parameter<double>(
    "pose_rebase_translation_threshold",
    readDouble(configuration, "ROS.pose_rebase_translation_threshold", 0.30));
  pose_rebase_rotation_threshold_ = declare_parameter<double>(
    "pose_rebase_rotation_threshold",
    readDouble(configuration, "ROS.pose_rebase_rotation_threshold", 0.45));

  if (output_stable_frames_required_ <= 0 ||
      pose_rebase_translation_threshold_ <= 0.0 ||
      pose_rebase_rotation_threshold_ <= 0.0)
  {
    throw std::runtime_error(
            "output stability and pose rebase parameters must be positive");
  }

  const bool enable_viewer = declare_parameter<bool>(
    "enable_viewer", readBool(configuration, "ROS.enable_viewer", false));

  const int image_width = declare_parameter<int>("image_width", 0);
  const int image_height = declare_parameter<int>("image_height", 0);
  const int image_fps = declare_parameter<int>("image_fps", 0);
  if (image_width <= 0 || image_height <= 0 || image_fps <= 0)
  {
    throw std::runtime_error(
            "parameters 'image_width', 'image_height' and 'image_fps' must be positive");
  }

  configuration.release();
  const std::string effective_settings = createEffectiveSettings(
    settings, image_width, image_height, image_fps);

  RCLCPP_INFO(get_logger(),
              "Loading ORB vocabulary: %s, settings: %s, image: %dx%d@%d, IMU: %s, "
              "Pangolin viewer: %s",
              vocabulary.c_str(), settings.c_str(),
              image_width, image_height, image_fps,
              use_imu_ ? "enabled" : "disabled",
              enable_viewer ? "enabled" : "disabled");

  // 2 initialize the ORB-SLAM3 system with the specified vocabulary, settings, and sensor type
  // 2.1 choose the sensor type based on whether IMU is used
  const auto sensor = use_imu_ ? ORB_SLAM3::System::IMU_RGBD : ORB_SLAM3::System::RGBD;
  // 2.2 start the ORB-SLAM3 system
  slam_ = std::make_unique<ORB_SLAM3::System>(
    vocabulary, effective_settings, sensor, enable_viewer);
  RCLCPP_INFO(get_logger(), "ORB-SLAM3 system initialized with sensor type: %s",
              sensor == ORB_SLAM3::System::IMU_RGBD ? "IMU_RGBD" : "RGBD");

  // 3 Topic subscription and distribution
  // 3.0 Create publishers for various topics
  pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("pose", 10);
  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
  path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 10);
  points_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("map_points", 2);
  state_pub_ = create_publisher<std_msgs::msg::Int32>("tracking_state", 10);
  imu_initialized_pub_ = create_publisher<std_msgs::msg::Bool>("imu_initialized", 10);
  imu_ba2_initialized_pub_ = create_publisher<std_msgs::msg::Bool>(
    "imu_ba2_initialized", 10);
  imu_samples_pub_ = create_publisher<std_msgs::msg::UInt32>("imu_samples_per_frame", 10);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // 3.1 Subscribe to color and depth topics and set up the synchronizer
  color_sub_.subscribe(this, color_topic_, rmw_qos_profile_sensor_data);
  depth_sub_.subscribe(this, depth_topic_, rmw_qos_profile_sensor_data);
  sync_ = std::make_shared<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image>>>(
    message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image,
      sensor_msgs::msg::Image>(20),
    color_sub_, depth_sub_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.035));      // Match the stable pre-5 ms RGB-D synchronization behavior
  sync_->registerCallback(        // Register it in the callback function to output the synchronized images
    std::bind(&OrbSlam3RgbdNode::imageCallback, this,
              std::placeholders::_1, std::placeholders::_2));

  // 3.2 Subscribe to IMU topic if IMU is used
  if (use_imu_)
  {
    // "Reentrant" means it can be scheduled by multiple threads
    imu_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    // bind the IMU callback to the subscription options, and belong to the reentrant callback group
    rclcpp::SubscriptionOptions imu_options;
    imu_options.callback_group = imu_callback_group_;
    // when imu_topic_ has messages, the imuCallback function will be called
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_, rclcpp::SensorDataQoS(),
      std::bind(&OrbSlam3RgbdNode::imuCallback, this, std::placeholders::_1), imu_options);     // bind the sub to the imuCallback
  }
  // 3.3 Initialize the path message header
  path_.header.frame_id = map_frame_;

  // debug info
  RCLCPP_INFO(get_logger(), "Waiting for RGB-D%s: %s + %s",
              use_imu_ ? " + IMU" : "", color_topic_.c_str(), depth_topic_.c_str());
  if (use_imu_)
  {
    RCLCPP_INFO(get_logger(), "IMU topic: %s", imu_topic_.c_str());
  }
}

OrbSlam3RgbdNode::~OrbSlam3RgbdNode()
{
  if (slam_)
  {
    slam_->Shutdown();
  }

  if (!effective_settings_path_.empty())
  {
    std::error_code error;
    std::filesystem::remove(effective_settings_path_, error);
  }
}

std::string OrbSlam3RgbdNode::readString(
  const cv::FileStorage &configuration,
  const std::string &key,
  const std::string &fallback)
{
  const cv::FileNode node = configuration[key];
  if (node.empty())
  {
    return fallback;
  }

  return static_cast<std::string>(node);
}

bool OrbSlam3RgbdNode::readBool(
  const cv::FileStorage &configuration,
  const std::string &key,
  bool fallback)
{
  const cv::FileNode node = configuration[key];
  if (node.empty())
  {
    return fallback;
  }

  return static_cast<int>(node) != 0;
}

int OrbSlam3RgbdNode::readInt(
  const cv::FileStorage &configuration,
  const std::string &key,
  int fallback)
{
  const cv::FileNode node = configuration[key];
  if (node.empty())
  {
    return fallback;
  }

  return static_cast<int>(node);
}

double OrbSlam3RgbdNode::readDouble(
  const cv::FileStorage &configuration,
  const std::string &key,
  double fallback)
{
  const cv::FileNode node = configuration[key];
  if (node.empty())
  {
    return fallback;
  }

  return static_cast<double>(node);
}

std::string OrbSlam3RgbdNode::createEffectiveSettings(
  const std::string &source_settings,
  int image_width,
  int image_height,
  int image_fps)
{
  effective_settings_path_ = std::filesystem::temp_directory_path() /
                             ("orb_slam3_effective_" + std::to_string(getpid()) + ".yaml");

  std::filesystem::copy_file(
    source_settings,
    effective_settings_path_,
    std::filesystem::copy_options::overwrite_existing);

  std::ofstream effective_settings(
    effective_settings_path_, std::ios::out | std::ios::app);
  if (!effective_settings)
  {
    throw std::runtime_error(
            "cannot create effective ORB-SLAM3 settings: " +
            effective_settings_path_.string());
  }

  effective_settings << "\nCamera.width: " << image_width << '\n';
  effective_settings << "Camera.height: " << image_height << '\n';
  effective_settings << "Camera.fps: " << image_fps << '\n';
  effective_settings.close();

  if (!effective_settings)
  {
    throw std::runtime_error(
            "cannot write effective ORB-SLAM3 settings: " +
            effective_settings_path_.string());
  }

  return effective_settings_path_.string();
}

void OrbSlam3RgbdNode::imuCallback(
  const sensor_msgs::msg::Imu::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(imu_mutex_);
  const double stamp = rclcpp::Time(msg->header.stamp).seconds();
  // "stamp <= imu_queue_.back().t" means the new input must be later than the last input in the queue
  // The IMUs in the queue must be strictly in chronological order.
  if (!imu_queue_.empty() && stamp <= imu_queue_.back().t)
  {
    ++imu_non_monotonic_;     // Abnormal situation stats
    return;
  }
  imu_queue_.emplace_back(      // imu queue's type is "std::deque<ORB_SLAM3::IMU::Point>"
    static_cast<float>(msg->linear_acceleration.x),
    static_cast<float>(msg->linear_acceleration.y),
    static_cast<float>(msg->linear_acceleration.z),
    static_cast<float>(msg->angular_velocity.x),
    static_cast<float>(msg->angular_velocity.y),
    static_cast<float>(msg->angular_velocity.z), stamp);

  // set the limit for the IMU queue size to prevent excessive memory usage
  while (imu_queue_.size() > 2000)
  {
    RCLCPP_WARN(get_logger(), "IMU queue size exceeds the limit(2000), removing old entries");
    imu_queue_.pop_front();
  }
}

std::vector<ORB_SLAM3::IMU::Point> OrbSlam3RgbdNode::takeImuUntil(
  double image_stamp)
{
  std::vector<ORB_SLAM3::IMU::Point> samples;
  if (!use_imu_)
  {
    return samples;
  }
  std::lock_guard<std::mutex> lock(imu_mutex_);
  while (!imu_queue_.empty() && imu_queue_.front().t <= image_stamp)
  {
    if (last_image_stamp_ < 0.0 || imu_queue_.front().t > last_image_stamp_)
    {
      samples.push_back(imu_queue_.front());
    }
    imu_queue_.pop_front();
  }
  return samples;
}

cv::Mat OrbSlam3RgbdNode::depthMeters(
  const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{

  if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
      msg->encoding == sensor_msgs::image_encodings::MONO16)
  {
    cv::Mat out;
    cv_bridge::toCvShare(msg, msg->encoding)->image.convertTo(out, CV_32F, 0.001);
    return out;
  }
  if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    return cv_bridge::toCvCopy(msg, msg->encoding)->image;
  }
  throw std::runtime_error("unsupported depth encoding: " + msg->encoding);
}

void OrbSlam3RgbdNode::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr &color_msg,
  const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg)
{
  std::lock_guard<std::mutex> lock(track_mutex_);
  try
  {
    // 1 change ros message to cv::Mat
    const cv::Mat color = cv_bridge::toCvCopy(
      color_msg, sensor_msgs::image_encodings::BGR8)->image;
    const cv::Mat depth = depthMeters(depth_msg);

    // 2 extract the timestamp and IMU samples for the current image
    const double stamp = rclcpp::Time(color_msg->header.stamp).seconds();
    const auto imu_samples = takeImuUntil(stamp);
    if (use_imu_ && imu_samples.size() < 2)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,      // it means printing on the termainal per 2000ms
                           "Waiting for at least two IMU samples before processing RGB-D");
      return;
    }

    // 3 set the input for the ORB_SLAM3 tracking module
    const Sophus::SE3f tcw = slam_->TrackRGBD(color, depth, stamp, imu_samples);    // tcw means the transformation from world to camera
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

    std_msgs::msg::Bool imu_ba2_initialized;
    imu_ba2_initialized.data = use_imu_ && slam_->IsImuBA2Initialized();
    imu_ba2_initialized_pub_->publish(imu_ba2_initialized);

    const bool tracking_ok = state.data == ORB_SLAM3::Tracking::OK;

    if (use_imu_ && !imu_initialized.data)
    {
      output_stable_frames_ = 0;
      if (output_ready_)
      {
        clearPublishedMap(
          color_msg->header.stamp,
          "IMU initialization lost or active SLAM map reset");
        output_ready_ = false;
      }
      return;
    }

    if (!tracking_ok)
    {
      if (!output_ready_)
      {
        output_stable_frames_ = 0;
      }
      return;
    }

    if (use_imu_ && !output_ready_)
    {
      ++output_stable_frames_;
      if (output_stable_frames_ < output_stable_frames_required_)
      {
        return;
      }

      output_ready_ = true;
      path_.poses.clear();
      output_from_orb_ = Sophus::SE3f();
      have_last_output_pose_ = false;
      RCLCPP_INFO(
        get_logger(),
        "IMU and tracking stable for %d frames; enabling continuous pose, path and map output",
        output_stable_frames_);
    }

    const Sophus::SE3f twc = continuousPose(tcw.inverse());
    publishPose(twc, color_msg->header.stamp);
    publishMapPoints(color_msg->header.stamp);
  }
  catch (const cv_bridge::Exception &e)
  {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                          "cv_bridge: %s", e.what());
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                          "RGB-D tracking: %s", e.what());
  }
}

void OrbSlam3RgbdNode::clearPublishedMap(
  const builtin_interfaces::msg::Time &stamp,
  const char *reason)
{
  path_.poses.clear();
  path_.header.stamp = stamp;
  path_pub_->publish(path_);

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp = stamp;
  cloud.header.frame_id = map_frame_;
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(0);
  points_pub_->publish(cloud);

  output_from_orb_ = Sophus::SE3f();
  have_last_output_pose_ = false;

  RCLCPP_WARN(get_logger(), "Cleared published path and sparse map: %s", reason);
}

Sophus::SE3f OrbSlam3RgbdNode::continuousPose(const Sophus::SE3f &twc)
{
  Sophus::SE3f output_pose = output_from_orb_ * twc;
  if (have_last_output_pose_)
  {
    const Sophus::SE3f delta = last_output_pose_.inverse() * output_pose;
    const double translation = delta.translation().norm();
    const double rotation = Eigen::AngleAxisf(delta.rotationMatrix()).angle();

    if (translation > pose_rebase_translation_threshold_ ||
        rotation > pose_rebase_rotation_threshold_)
    {
      output_from_orb_ = last_output_pose_ * twc.inverse();
      output_pose = output_from_orb_ * twc;
      ++pose_rebase_count_;
      RCLCPP_WARN(
        get_logger(),
        "Rebased ORB world frame to preserve continuous output pose: "
        "translation %.3f m, rotation %.3f rad, count %lu",
        translation, rotation, static_cast<unsigned long>(pose_rebase_count_));
    }
  }

  last_output_pose_ = output_pose;
  have_last_output_pose_ = true;
  return output_pose;
}

geometry_msgs::msg::Pose OrbSlam3RgbdNode::poseFrom(
  const Sophus::SE3f &twc) const
{
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

void OrbSlam3RgbdNode::publishPose(
  const Sophus::SE3f &twc,
  const builtin_interfaces::msg::Time &stamp)
{
  // publish the pose
  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = map_frame_;
  pose.pose = poseFrom(twc);
  pose_pub_->publish(pose);

  // publish the odom
  nav_msgs::msg::Odometry odom;
  odom.header = pose.header;
  odom.child_frame_id = camera_frame_;
  odom.pose.pose = pose.pose;
  odom_pub_->publish(odom);

  // publish the path
  path_.header.stamp = stamp;
  path_.poses.push_back(pose);
  if (max_path_poses_ > 0 && static_cast<int>(path_.poses.size()) > max_path_poses_)
  {
    path_.poses.erase(path_.poses.begin(),
                      path_.poses.begin() + (path_.poses.size() - max_path_poses_));
  }
  path_pub_->publish(path_);

  // publish the TF
  geometry_msgs::msg::TransformStamped tf;
  tf.header = pose.header;
  tf.child_frame_id = camera_frame_;
  tf.transform.translation.x = pose.pose.position.x;
  tf.transform.translation.y = pose.pose.position.y;
  tf.transform.translation.z = pose.pose.position.z;
  tf.transform.rotation = pose.pose.orientation;
  tf_broadcaster_->sendTransform(tf);
}

void OrbSlam3RgbdNode::publishMapPoints(
  const builtin_interfaces::msg::Time &stamp)
{
  const auto map_points = slam_->GetAllMapPoints();
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp = stamp;
  cloud.header.frame_id = map_frame_;
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  std::size_t valid = 0;
  for (auto *point : map_points)
  {
    if (point != nullptr && !point->isBad())
    {
      ++valid;
    }
  }
  modifier.resize(valid);
  sensor_msgs::PointCloud2Iterator<float> x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> z(cloud, "z");
  for (auto *point : map_points)
  {
    if (point == nullptr || point->isBad())
    {
      continue;
    }
    const Eigen::Vector3f p = output_from_orb_ * point->GetWorldPos();
    *x = p.x(); *y = p.y(); *z = p.z();
    ++x; ++y; ++z;
  }
  points_pub_->publish(cloud);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);   // initialize the ROS 2 context, enter the ros2 management
  try
  {
    // create the node first
    auto node = std::make_shared<OrbSlam3RgbdNode>();
    // create the exector with multiple threads
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    // add the node to the executor, this mean connect the new node to ROS2
    executor.add_node(node);
    // start the executor, which will handle the ROS 2 callbacks and node operations
    executor.spin();
  }
  catch (const std::exception &e)
  {
    std::cerr << "in rgbd_node.cpp-main()" << std::endl;
    std::cerr << "in rgbd_node ORB-SLAM3 ROS 2 fatal error: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
