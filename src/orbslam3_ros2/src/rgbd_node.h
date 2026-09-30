#ifndef ORBSLAM3_ROS2__RGBD_NODE_H_
#define ORBSLAM3_ROS2__RGBD_NODE_H_

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "System.h"

class OrbSlam3RgbdNode final : public rclcpp::Node
{
 public:
  // 初始化参数、ORB-SLAM3、订阅器、发布器和 RGB-D 同步器。
  OrbSlam3RgbdNode();

  // 安全关闭 ORB-SLAM3，并删除运行时生成的临时配置。
 ~OrbSlam3RgbdNode() override;

 private:
  // 从 OpenCV YAML 读取字符串；字段不存在时返回默认值。
  static std::string readString(
    const cv::FileStorage &configuration,
    const std::string &key,
    const std::string &fallback);

  // 从 OpenCV YAML 读取布尔值；字段不存在时返回默认值。
  static bool readBool(
    const cv::FileStorage &configuration,
    const std::string &key,
    bool fallback);

  // 从 OpenCV YAML 读取整数；字段不存在时返回默认值。
  static int readInt(
    const cv::FileStorage &configuration,
    const std::string &key,
    int fallback);

  // 从 OpenCV YAML 读取浮点数；字段不存在时返回默认值。
  static double readDouble(
    const cv::FileStorage &configuration,
    const std::string &key,
    double fallback);

  // 复制 SLAM 配置，并注入相机 YAML 中统一管理的图像尺寸和帧率。
  std::string createEffectiveSettings(
    const std::string &source_settings,
    int image_width,
    int image_height,
    int image_fps);

  // 接收 IMU 消息，并按时间顺序缓存为 ORB-SLAM3 IMU 样本。
  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);

  // 取出当前图像时间戳之前且尚未使用的 IMU 样本。
  std::vector<ORB_SLAM3::IMU::Point> takeImuUntil(double image_stamp);

  // 将 ROS 深度图转换为以米为单位的单通道浮点图像。
  static cv::Mat depthMeters(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg);

  // 处理完成软同步的彩色图和深度图，并调用 ORB-SLAM3 跟踪。
  void imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &color_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg);

  // 清空不再属于当前稳定惯性地图的轨迹和稀疏点云显示。
  void clearPublishedMap(
    const builtin_interfaces::msg::Time &stamp,
    const char *reason);

  // 补偿 ORB 世界坐标的突变，生成对外连续的相机位姿。
  Sophus::SE3f continuousPose(const Sophus::SE3f &twc);

  // 将 ORB-SLAM3 相机位姿转换为 ROS Pose 消息。
  geometry_msgs::msg::Pose poseFrom(const Sophus::SE3f &twc) const;

  // 发布相机位姿、里程计、轨迹和 TF。
  void publishPose(
    const Sophus::SE3f &twc,
    const builtin_interfaces::msg::Time &stamp);

  // 发布当前 Atlas 中所有有效地图点。
  void publishMapPoints(const builtin_interfaces::msg::Time &stamp);

  std::unique_ptr<ORB_SLAM3::System> slam_;
  std::filesystem::path effective_settings_path_;
  message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer
      <message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,sensor_msgs::msg::Image>>> sync_;
  std::mutex track_mutex_;
  std::mutex imu_mutex_;
  std::deque<ORB_SLAM3::IMU::Point> imu_queue_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;
  std::string color_topic_;
  std::string depth_topic_;
  std::string imu_topic_;
  std::string map_frame_;
  std::string camera_frame_;
  bool use_imu_
  {
    false
  }
  ;
  double last_image_stamp_
  {
    -1.0
  }
  ;
  uint64_t imu_non_monotonic_
  {
    0
  }
  ;
  int max_path_poses_;
  int output_stable_frames_required_
  {
    30
  }
  ;
  int output_stable_frames_
  {
    0
  }
  ;
  bool output_ready_
  {
    false
  }
  ;
  double pose_rebase_translation_threshold_
  {
    0.30
  }
  ;
  double pose_rebase_rotation_threshold_
  {
    0.45
  }
  ;
  Sophus::SE3f output_from_orb_;
  Sophus::SE3f last_output_pose_;
  bool have_last_output_pose_
  {
    false
  }
  ;
  uint64_t pose_rebase_count_
  {
    0
  }
  ;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr state_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr imu_initialized_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr imu_ba2_initialized_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr imu_samples_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  nav_msgs::msg::Path path_;
}
;

#endif  // ORBSLAM3_ROS2__RGBD_NODE_H_
