#ifndef ORBSLAM3_ROS2__OCCUPANCY_MAPPER_H_
#define ORBSLAM3_ROS2__OCCUPANCY_MAPPER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

class OccupancyMapper : public rclcpp::Node
{
public:
  // Initializes the lightweight depth-to-occupancy mapping node.
  OccupancyMapper();

private:
  // Declares and reads all mapping parameters.
  void readParameters();

  // Validates parameters and allocates the fixed-size grid buffers.
  void initializeGrid();

  // Creates topic subscriptions, synchronization, publishers, and services.
  void initializeRosInterfaces();

  // Updates the camera intrinsic parameters used for depth projection.
  void cameraInfoCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr message);

  // Precomputes distortion-corrected normalized camera rays for every pixel.
  void buildProjectionRays(const sensor_msgs::msg::CameraInfo &message);

  // Updates whether ORB-SLAM3 is currently tracking normally.
  void trackingStateCallback(std_msgs::msg::Int32::ConstSharedPtr message);

  // Updates inertial readiness and clears the grid after an inertial reset.
  void imuInitializedCallback(std_msgs::msg::Bool::ConstSharedPtr message);

  // Processes a synchronized depth image and ORB camera pose.
  void synchronizedCallback(
    sensor_msgs::msg::Image::ConstSharedPtr depth_message,
    geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_message);

  // Converts supported ROS depth encodings into a meter-valued float image.
  cv::Mat depthMeters(sensor_msgs::msg::Image::ConstSharedPtr message) const;

  // Initializes the gravity-aligned navigation frame from the first valid pose.
  bool initializeNavigationFrame(
    const geometry_msgs::msg::PoseStamped &pose_message,
    const std::array<double, 9> &camera_rotation);

  // Returns the rotation matrix represented by a normalized quaternion.
  std::array<double, 9> rotationMatrix(
    const geometry_msgs::msg::Quaternion &quaternion) const;

  // Detects an unsafe discontinuity in the published ORB camera pose.
  bool poseIsDiscontinuous(const geometry_msgs::msg::Pose &pose) const;

  // Stores the pose used for the next discontinuity check.
  void updateLastPose(const geometry_msgs::msg::Pose &pose);

  // Projects one depth pixel into the gravity-aligned navigation frame.
  void projectToNavigationFrame(
    int u,
    int v,
    float depth,
    const std::array<double, 9> &camera_rotation,
    const geometry_msgs::msg::Point &camera_position,
    double &navigation_x,
    double &navigation_y,
    double &navigation_z) const;

  // Converts a navigation-frame metric position into a grid cell index.
  bool gridIndex(double x, double y, int &grid_x, int &grid_y) const;

  // Marks free cells along one sensor ray and optionally its occupied endpoint.
  void markRay(
    int start_x,
    int start_y,
    int end_x,
    int end_y,
    bool occupied_endpoint);

  // Applies one evidence update per touched cell for the current depth frame.
  void applyFrameUpdates();

  // Publishes the current occupancy grid and metadata.
  void publishMap(const builtin_interfaces::msg::Time &stamp);

  // Publishes the fixed gravity-aligned transform from orb_map to nav_map.
  void publishNavigationTransform(const builtin_interfaces::msg::Time &stamp);

  // Clears occupancy evidence while preserving the current navigation frame.
  void clearGrid(const char *reason);

  // Clears occupancy evidence and waits for a new navigation-frame origin.
  void resetMapper(const char *reason);

  // Handles a manual request to clear the current grid.
  void clearServiceCallback(
    const std_srvs::srv::Empty::Request::SharedPtr request,
    std_srvs::srv::Empty::Response::SharedPtr response);

  // Prints compact processing and map-occupancy statistics.
  void logStatistics();

  double resolution_ = 0.05;
  double map_width_meters_ = 30.0;
  double map_height_meters_ = 30.0;
  double sync_max_interval_ms_ = 10.0;
  double min_depth_ = 0.35;
  double max_depth_ = 5.0;
  double initial_camera_height_ = 1.0;
  double obstacle_min_height_ = 0.10;
  double obstacle_max_height_ = 1.80;
  double floor_clearance_height_ = 0.15;
  double pose_jump_translation_threshold_ = 0.30;
  double pose_jump_rotation_threshold_ = 0.45;

  int pixel_stride_ = 8;
  int process_every_n_frames_ = 6;
  int publish_every_n_processed_frames_ = 5;
  int hit_log_odds_ = 4;
  int miss_log_odds_ = 1;
  int min_log_odds_ = -20;
  int max_log_odds_ = 20;
  int occupied_log_odds_threshold_ = 2;
  int free_log_odds_threshold_ = -1;
  int map_width_cells_ = 0;
  int map_height_cells_ = 0;

  bool use_imu_ = true;
  bool tracking_ok_ = false;
  bool imu_initialized_ = false;
  bool navigation_frame_initialized_ = false;
  bool have_last_pose_ = false;

  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  int calibration_width_ = 0;
  int calibration_height_ = 0;

  double navigation_origin_world_x_ = 0.0;
  double navigation_origin_world_y_ = 0.0;
  double navigation_origin_world_z_ = 0.0;
  double navigation_yaw_ = 0.0;
  double navigation_yaw_cosine_ = 1.0;
  double navigation_yaw_sine_ = 0.0;

  std::array<double, 3> last_position_{};
  std::array<double, 4> last_orientation_{};

  std::vector<int16_t> log_odds_grid_;
  std::vector<uint8_t> observed_grid_;
  std::vector<uint8_t> frame_free_cells_;
  std::vector<uint8_t> frame_occupied_cells_;
  std::vector<float> projection_ray_x_;
  std::vector<float> projection_ray_y_;

  uint64_t synchronized_frames_ = 0;
  uint64_t processed_frames_ = 0;
  uint64_t valid_depth_samples_ = 0;
  uint64_t obstacle_samples_ = 0;
  uint64_t floor_samples_ = 0;

  builtin_interfaces::msg::Time map_load_time_;
  builtin_interfaces::msg::Time last_message_stamp_;

  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string pose_topic_;
  std::string tracking_state_topic_;
  std::string imu_initialized_topic_;
  std::string world_frame_;
  std::string map_frame_;

  std::mutex calibration_mutex_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr tracking_state_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr imu_initialized_subscription_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;
  rclcpp::Publisher<nav_msgs::msg::MapMetaData>::SharedPtr metadata_publisher_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr clear_service_;
  rclcpp::TimerBase::SharedPtr statistics_timer_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> transform_broadcaster_;

  message_filters::Subscriber<sensor_msgs::msg::Image> depth_subscription_;
  message_filters::Subscriber<geometry_msgs::msg::PoseStamped> pose_subscription_;
  std::shared_ptr<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        geometry_msgs::msg::PoseStamped>>> synchronizer_;
};

#endif  // ORBSLAM3_ROS2__OCCUPANCY_MAPPER_H_
