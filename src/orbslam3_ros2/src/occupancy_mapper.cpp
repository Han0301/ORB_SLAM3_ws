#include "occupancy_mapper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>

#include <opencv2/calib3d.hpp>
#include <sensor_msgs/image_encodings.hpp>

// Initializes the lightweight depth-to-occupancy mapping node.
OccupancyMapper::OccupancyMapper()
: Node("occupancy_mapper")
{
  readParameters();
  initializeGrid();
  initializeRosInterfaces();

  RCLCPP_INFO(
    get_logger(),
    "Occupancy mapper ready: %.2f x %.2f m, %.3f m/cell, depth %.2f-%.2f m, "
    "stride %d, processing %.1f Hz from a 30 Hz input",
    map_width_meters_,
    map_height_meters_,
    resolution_,
    min_depth_,
    max_depth_,
    pixel_stride_,
    30.0 / static_cast<double>(process_every_n_frames_));
}

// Declares and reads all mapping parameters.
void OccupancyMapper::readParameters()
{
  resolution_ = declare_parameter<double>("resolution", 0.05);
  map_width_meters_ = declare_parameter<double>("map_width", 30.0);
  map_height_meters_ = declare_parameter<double>("map_height", 30.0);

  pixel_stride_ = declare_parameter<int>("pixel_stride", 8);
  process_every_n_frames_ = declare_parameter<int>("process_every_n_frames", 6);
  publish_every_n_processed_frames_ =
    declare_parameter<int>("publish_every_n_processed_frames", 5);
  sync_max_interval_ms_ = declare_parameter<double>("sync_max_interval_ms", 10.0);

  min_depth_ = declare_parameter<double>("min_depth", 0.35);
  max_depth_ = declare_parameter<double>("max_depth", 5.0);

  initial_camera_height_ = declare_parameter<double>("initial_camera_height", 1.0);
  obstacle_min_height_ = declare_parameter<double>("obstacle_min_height", 0.10);
  obstacle_max_height_ = declare_parameter<double>("obstacle_max_height", 1.80);
  floor_clearance_height_ = declare_parameter<double>("floor_clearance_height", 0.15);

  hit_log_odds_ = declare_parameter<int>("hit_log_odds", 4);
  miss_log_odds_ = declare_parameter<int>("miss_log_odds", 1);
  min_log_odds_ = declare_parameter<int>("min_log_odds", -20);
  max_log_odds_ = declare_parameter<int>("max_log_odds", 20);
  occupied_log_odds_threshold_ =
    declare_parameter<int>("occupied_log_odds_threshold", 2);
  free_log_odds_threshold_ = declare_parameter<int>("free_log_odds_threshold", -1);

  pose_jump_translation_threshold_ =
    declare_parameter<double>("pose_jump_translation_threshold", 0.30);
  pose_jump_rotation_threshold_ =
    declare_parameter<double>("pose_jump_rotation_threshold", 0.45);

  use_imu_ = declare_parameter<bool>("use_imu", true);
  depth_topic_ = declare_parameter<std::string>(
    "depth_topic", "/camera/depth/image_raw");
  camera_info_topic_ = declare_parameter<std::string>(
    "camera_info_topic", "/camera/color/camera_info");
  pose_topic_ = declare_parameter<std::string>(
    "pose_topic", "/orb_slam3/pose");
  tracking_state_topic_ = declare_parameter<std::string>(
    "tracking_state_topic", "/orb_slam3/tracking_state");
  imu_initialized_topic_ = declare_parameter<std::string>(
    "imu_initialized_topic", "/orb_slam3/imu_initialized");
  world_frame_ = declare_parameter<std::string>("world_frame", "orb_map");
  map_frame_ = declare_parameter<std::string>("map_frame", "nav_map");
}

// Validates parameters and allocates the fixed-size grid buffers.
void OccupancyMapper::initializeGrid()
{
  if (resolution_ <= 0.0 || map_width_meters_ <= 0.0 || map_height_meters_ <= 0.0)
  {
    throw std::runtime_error("map resolution and dimensions must be positive");
  }

  if (pixel_stride_ <= 0 || process_every_n_frames_ <= 0 ||
      publish_every_n_processed_frames_ <= 0)
  {
    throw std::runtime_error("frame and pixel stride parameters must be positive");
  }

  if (min_depth_ <= 0.0 || max_depth_ <= min_depth_ ||
      initial_camera_height_ <= 0.0)
  {
    throw std::runtime_error("depth range and initial camera height are invalid");
  }

  if (obstacle_min_height_ < 0.0 ||
      obstacle_max_height_ <= obstacle_min_height_ ||
      floor_clearance_height_ < 0.0)
  {
    throw std::runtime_error("height-filter parameters are invalid");
  }

  if (min_log_odds_ >= max_log_odds_ || hit_log_odds_ <= 0 || miss_log_odds_ <= 0)
  {
    throw std::runtime_error("log-odds parameters are invalid");
  }

  map_width_cells_ = static_cast<int>(std::ceil(map_width_meters_ / resolution_));
  map_height_cells_ = static_cast<int>(std::ceil(map_height_meters_ / resolution_));
  map_width_meters_ = static_cast<double>(map_width_cells_) * resolution_;
  map_height_meters_ = static_cast<double>(map_height_cells_) * resolution_;

  const std::size_t cell_count =
    static_cast<std::size_t>(map_width_cells_) * static_cast<std::size_t>(map_height_cells_);
  log_odds_grid_.assign(cell_count, 0);
  observed_grid_.assign(cell_count, 0);
  frame_free_cells_.assign(cell_count, 0);
  frame_occupied_cells_.assign(cell_count, 0);
}

// Creates topic subscriptions, synchronization, publishers, and services.
void OccupancyMapper::initializeRosInterfaces()
{
  camera_info_subscription_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&OccupancyMapper::cameraInfoCallback, this, std::placeholders::_1));

  tracking_state_subscription_ = create_subscription<std_msgs::msg::Int32>(
    tracking_state_topic_,
    10,
    std::bind(&OccupancyMapper::trackingStateCallback, this, std::placeholders::_1));

  imu_initialized_subscription_ = create_subscription<std_msgs::msg::Bool>(
    imu_initialized_topic_,
    10,
    std::bind(&OccupancyMapper::imuInitializedCallback, this, std::placeholders::_1));

  depth_subscription_.subscribe(this, depth_topic_, rmw_qos_profile_sensor_data);
  pose_subscription_.subscribe(this, pose_topic_, rmw_qos_profile_sensor_data);

  synchronizer_ = std::make_shared<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        geometry_msgs::msg::PoseStamped>>>(
    message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image,
      geometry_msgs::msg::PoseStamped>(30),
    depth_subscription_,
    pose_subscription_);
  synchronizer_->setMaxIntervalDuration(
    rclcpp::Duration::from_seconds(sync_max_interval_ms_ / 1000.0));
  synchronizer_->registerCallback(
    std::bind(
      &OccupancyMapper::synchronizedCallback,
      this,
      std::placeholders::_1,
      std::placeholders::_2));

  const rclcpp::QoS map_qos = rclcpp::QoS(1).reliable().transient_local();
  map_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>("map", map_qos);
  metadata_publisher_ = create_publisher<nav_msgs::msg::MapMetaData>("map_metadata", map_qos);

  clear_service_ = create_service<std_srvs::srv::Empty>(
    "~/clear",
    std::bind(
      &OccupancyMapper::clearServiceCallback,
      this,
      std::placeholders::_1,
      std::placeholders::_2));

  transform_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
  statistics_timer_ = create_wall_timer(
    std::chrono::seconds(10),
    std::bind(&OccupancyMapper::logStatistics, this));
}

// Updates the camera intrinsic parameters used for depth projection.
void OccupancyMapper::cameraInfoCallback(
  sensor_msgs::msg::CameraInfo::ConstSharedPtr message)
{
  std::lock_guard<std::mutex> lock(calibration_mutex_);
  if (!projection_ray_x_.empty() &&
      calibration_width_ == static_cast<int>(message->width) &&
      calibration_height_ == static_cast<int>(message->height) &&
      std::abs(fx_ - message->k[0]) < 1.0e-9 &&
      std::abs(fy_ - message->k[4]) < 1.0e-9 &&
      std::abs(cx_ - message->k[2]) < 1.0e-9 &&
      std::abs(cy_ - message->k[5]) < 1.0e-9)
  {
    return;
  }

  fx_ = message->k[0];
  fy_ = message->k[4];
  cx_ = message->k[2];
  cy_ = message->k[5];
  buildProjectionRays(*message);
}

// Precomputes distortion-corrected normalized camera rays for every pixel.
void OccupancyMapper::buildProjectionRays(const sensor_msgs::msg::CameraInfo &message)
{
  calibration_width_ = static_cast<int>(message.width);
  calibration_height_ = static_cast<int>(message.height);
  const std::size_t pixel_count =
    static_cast<std::size_t>(calibration_width_) *
    static_cast<std::size_t>(calibration_height_);

  std::vector<cv::Point2f> distorted_pixels(pixel_count);
  for (int v = 0; v < calibration_height_; ++v)
  {
    for (int u = 0; u < calibration_width_; ++u)
    {
      const std::size_t index =
        static_cast<std::size_t>(v) * static_cast<std::size_t>(calibration_width_) +
        static_cast<std::size_t>(u);
      distorted_pixels[index] = cv::Point2f(
        static_cast<float>(u), static_cast<float>(v));
    }
  }

  std::vector<cv::Point2f> normalized_pixels;
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
    fx_, 0.0, cx_,
    0.0, fy_, cy_,
    0.0, 0.0, 1.0);
  const cv::Mat distortion_coefficients(message.d, true);

  if (message.d.empty())
  {
    normalized_pixels.resize(pixel_count);
    for (std::size_t index = 0; index < pixel_count; ++index)
    {
      normalized_pixels[index].x =
        (distorted_pixels[index].x - static_cast<float>(cx_)) /
        static_cast<float>(fx_);
      normalized_pixels[index].y =
        (distorted_pixels[index].y - static_cast<float>(cy_)) /
        static_cast<float>(fy_);
    }
  }
  else
  {
    cv::undistortPoints(
      distorted_pixels,
      normalized_pixels,
      camera_matrix,
      distortion_coefficients);
  }

  projection_ray_x_.resize(pixel_count);
  projection_ray_y_.resize(pixel_count);
  for (std::size_t index = 0; index < pixel_count; ++index)
  {
    projection_ray_x_[index] = normalized_pixels[index].x;
    projection_ray_y_[index] = normalized_pixels[index].y;
  }

  RCLCPP_INFO(
    get_logger(),
    "Projection rays ready: %dx%d, distortion coefficients=%zu",
    calibration_width_, calibration_height_, message.d.size());
}

// Updates whether ORB-SLAM3 is currently tracking normally.
void OccupancyMapper::trackingStateCallback(std_msgs::msg::Int32::ConstSharedPtr message)
{
  tracking_ok_ = message->data == 2;
}

// Updates inertial readiness and clears the grid after an inertial reset.
void OccupancyMapper::imuInitializedCallback(std_msgs::msg::Bool::ConstSharedPtr message)
{
  const bool was_initialized = imu_initialized_;
  imu_initialized_ = message->data;

  if (use_imu_ && was_initialized && !imu_initialized_)
  {
    resetMapper("IMU initialization lost or active ORB map reset");
  }
}

// Processes a synchronized depth image and ORB camera pose.
void OccupancyMapper::synchronizedCallback(
  sensor_msgs::msg::Image::ConstSharedPtr depth_message,
  geometry_msgs::msg::PoseStamped::ConstSharedPtr pose_message)
{
  ++synchronized_frames_;
  last_message_stamp_ = depth_message->header.stamp;

  if (!tracking_ok_ || (use_imu_ && !imu_initialized_))
  {
    return;
  }

  if ((synchronized_frames_ - 1) % static_cast<uint64_t>(process_every_n_frames_) != 0)
  {
    return;
  }

  double fx = 0.0;
  double fy = 0.0;
  {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    fx = fx_;
    fy = fy_;
  }
  if (fx <= 0.0 || fy <= 0.0)
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Waiting for valid color-camera intrinsics");
    return;
  }

  try
  {
    const cv::Mat depth = depthMeters(depth_message);
    const std::array<double, 9> camera_rotation =
      rotationMatrix(pose_message->pose.orientation);

    if (!navigation_frame_initialized_)
    {
      if (!initializeNavigationFrame(*pose_message, camera_rotation))
      {
        return;
      }
    }

    if (poseIsDiscontinuous(pose_message->pose))
    {
      resetMapper("discontinuous ORB camera pose");
      return;
    }
    updateLastPose(pose_message->pose);

    const double camera_world_dx =
      pose_message->pose.position.x - navigation_origin_world_x_;
    const double camera_world_dy =
      pose_message->pose.position.y - navigation_origin_world_y_;
    const double camera_navigation_x =
      navigation_yaw_cosine_ * camera_world_dx +
      navigation_yaw_sine_ * camera_world_dy;
    const double camera_navigation_y =
      -navigation_yaw_sine_ * camera_world_dx +
      navigation_yaw_cosine_ * camera_world_dy;

    int camera_grid_x = 0;
    int camera_grid_y = 0;
    if (!gridIndex(
        camera_navigation_x,
        camera_navigation_y,
        camera_grid_x,
        camera_grid_y))
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Camera left the fixed %.1f x %.1f m occupancy grid",
        map_width_meters_, map_height_meters_);
      return;
    }

    std::fill(frame_free_cells_.begin(), frame_free_cells_.end(), 0);
    std::fill(frame_occupied_cells_.begin(), frame_occupied_cells_.end(), 0);

    for (int v = 0; v < depth.rows; v += pixel_stride_)
    {
      for (int u = 0; u < depth.cols; u += pixel_stride_)
      {
        const float range = depth.at<float>(v, u);
        if (!std::isfinite(range) || range < min_depth_ || range > max_depth_)
        {
          continue;
        }

        ++valid_depth_samples_;
        double navigation_x = 0.0;
        double navigation_y = 0.0;
        double navigation_z = 0.0;
        projectToNavigationFrame(
          u,
          v,
          range,
          camera_rotation,
          pose_message->pose.position,
          navigation_x,
          navigation_y,
          navigation_z);

        int endpoint_grid_x = 0;
        int endpoint_grid_y = 0;
        if (!gridIndex(
            navigation_x,
            navigation_y,
            endpoint_grid_x,
            endpoint_grid_y))
        {
          continue;
        }

        if (navigation_z >= obstacle_min_height_ &&
            navigation_z <= obstacle_max_height_)
        {
          ++obstacle_samples_;
          markRay(
            camera_grid_x,
            camera_grid_y,
            endpoint_grid_x,
            endpoint_grid_y,
            true);
        }
        else if (navigation_z >= -floor_clearance_height_ &&
                 navigation_z < obstacle_min_height_)
        {
          ++floor_samples_;
          markRay(
            camera_grid_x,
            camera_grid_y,
            endpoint_grid_x,
            endpoint_grid_y,
            false);
        }
      }
    }

    applyFrameUpdates();
    ++processed_frames_;

    if (processed_frames_ % static_cast<uint64_t>(publish_every_n_processed_frames_) == 0)
    {
      publishMap(depth_message->header.stamp);
    }
  }
  catch (const cv_bridge::Exception &exception)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Depth conversion failed: %s", exception.what());
  }
  catch (const std::exception &exception)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Occupancy update failed: %s", exception.what());
  }
}

// Converts supported ROS depth encodings into a meter-valued float image.
cv::Mat OccupancyMapper::depthMeters(
  sensor_msgs::msg::Image::ConstSharedPtr message) const
{
  if (message->encoding == sensor_msgs::image_encodings::TYPE_16UC1 ||
      message->encoding == sensor_msgs::image_encodings::MONO16)
  {
    cv::Mat depth;
    cv_bridge::toCvShare(message, message->encoding)->image.convertTo(
      depth, CV_32F, 0.001);
    return depth;
  }

  if (message->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    return cv_bridge::toCvShare(message, message->encoding)->image;
  }

  throw std::runtime_error("unsupported depth encoding: " + message->encoding);
}

// Initializes the gravity-aligned navigation frame from the first valid pose.
bool OccupancyMapper::initializeNavigationFrame(
  const geometry_msgs::msg::PoseStamped &pose_message,
  const std::array<double, 9> &camera_rotation)
{
  const double forward_x = camera_rotation[2];
  const double forward_y = camera_rotation[5];
  const double horizontal_norm = std::hypot(forward_x, forward_y);
  if (horizontal_norm < 0.20)
  {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Cannot initialize nav_map while the camera points nearly vertically");
    return false;
  }

  navigation_yaw_ = std::atan2(forward_y, forward_x);
  navigation_yaw_cosine_ = std::cos(navigation_yaw_);
  navigation_yaw_sine_ = std::sin(navigation_yaw_);
  navigation_origin_world_x_ = pose_message.pose.position.x;
  navigation_origin_world_y_ = pose_message.pose.position.y;
  navigation_origin_world_z_ =
    pose_message.pose.position.z - initial_camera_height_;
  map_load_time_ = pose_message.header.stamp;
  navigation_frame_initialized_ = true;
  have_last_pose_ = false;

  publishNavigationTransform(pose_message.header.stamp);
  publishMap(pose_message.header.stamp);

  RCLCPP_INFO(
    get_logger(),
    "Initialized %s at floor z %.3f m with initial camera height %.3f m",
    map_frame_.c_str(), navigation_origin_world_z_, initial_camera_height_);
  return true;
}

// Returns the rotation matrix represented by a normalized quaternion.
std::array<double, 9> OccupancyMapper::rotationMatrix(
  const geometry_msgs::msg::Quaternion &quaternion) const
{
  const double norm = std::sqrt(
    quaternion.x * quaternion.x + quaternion.y * quaternion.y +
    quaternion.z * quaternion.z + quaternion.w * quaternion.w);
  if (norm < 1.0e-9)
  {
    return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  }

  const double x = quaternion.x / norm;
  const double y = quaternion.y / norm;
  const double z = quaternion.z / norm;
  const double w = quaternion.w / norm;
  const double xx = x * x;
  const double yy = y * y;
  const double zz = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double wx = w * x;
  const double wy = w * y;
  const double wz = w * z;

  return {
    1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
    2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
    2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)};
}

// Detects an unsafe discontinuity in the published ORB camera pose.
bool OccupancyMapper::poseIsDiscontinuous(const geometry_msgs::msg::Pose &pose) const
{
  if (!have_last_pose_)
  {
    return false;
  }

  const double dx = pose.position.x - last_position_[0];
  const double dy = pose.position.y - last_position_[1];
  const double dz = pose.position.z - last_position_[2];
  const double translation = std::sqrt(dx * dx + dy * dy + dz * dz);

  const double quaternion_dot = std::abs(
    pose.orientation.x * last_orientation_[0] +
    pose.orientation.y * last_orientation_[1] +
    pose.orientation.z * last_orientation_[2] +
    pose.orientation.w * last_orientation_[3]);
  const double rotation = 2.0 * std::acos(std::min(1.0, quaternion_dot));

  if (translation > pose_jump_translation_threshold_ ||
      rotation > pose_jump_rotation_threshold_)
  {
    RCLCPP_WARN(
      get_logger(),
      "Pose discontinuity: translation %.3f m, rotation %.3f rad",
      translation, rotation);
    return true;
  }

  return false;
}

// Stores the pose used for the next discontinuity check.
void OccupancyMapper::updateLastPose(const geometry_msgs::msg::Pose &pose)
{
  last_position_ = {pose.position.x, pose.position.y, pose.position.z};
  last_orientation_ = {
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z,
    pose.orientation.w};
  have_last_pose_ = true;
}

// Projects one depth pixel into the gravity-aligned navigation frame.
void OccupancyMapper::projectToNavigationFrame(
  int u,
  int v,
  float depth,
  const std::array<double, 9> &camera_rotation,
  const geometry_msgs::msg::Point &camera_position,
  double &navigation_x,
  double &navigation_y,
  double &navigation_z) const
{
  const std::size_t ray_index =
    static_cast<std::size_t>(v) * static_cast<std::size_t>(calibration_width_) +
    static_cast<std::size_t>(u);
  const bool ray_is_available =
    calibration_width_ > 0 && calibration_height_ > 0 &&
    u >= 0 && u < calibration_width_ && v >= 0 && v < calibration_height_ &&
    ray_index < projection_ray_x_.size();
  const double camera_ray_x = ray_is_available ?
    projection_ray_x_[ray_index] :
    (static_cast<double>(u) - cx_) / fx_;
  const double camera_ray_y = ray_is_available ?
    projection_ray_y_[ray_index] :
    (static_cast<double>(v) - cy_) / fy_;
  const double camera_x = camera_ray_x * depth;
  const double camera_y = camera_ray_y * depth;
  const double camera_z = depth;

  const double world_x =
    camera_rotation[0] * camera_x +
    camera_rotation[1] * camera_y +
    camera_rotation[2] * camera_z + camera_position.x;
  const double world_y =
    camera_rotation[3] * camera_x +
    camera_rotation[4] * camera_y +
    camera_rotation[5] * camera_z + camera_position.y;
  const double world_z =
    camera_rotation[6] * camera_x +
    camera_rotation[7] * camera_y +
    camera_rotation[8] * camera_z + camera_position.z;

  const double world_dx = world_x - navigation_origin_world_x_;
  const double world_dy = world_y - navigation_origin_world_y_;
  navigation_x =
    navigation_yaw_cosine_ * world_dx + navigation_yaw_sine_ * world_dy;
  navigation_y =
    -navigation_yaw_sine_ * world_dx + navigation_yaw_cosine_ * world_dy;
  navigation_z = world_z - navigation_origin_world_z_;
}

// Converts a navigation-frame metric position into a grid cell index.
bool OccupancyMapper::gridIndex(
  double x,
  double y,
  int &grid_x,
  int &grid_y) const
{
  grid_x = static_cast<int>(std::floor((x + map_width_meters_ * 0.5) / resolution_));
  grid_y = static_cast<int>(std::floor((y + map_height_meters_ * 0.5) / resolution_));
  return grid_x >= 0 && grid_x < map_width_cells_ &&
         grid_y >= 0 && grid_y < map_height_cells_;
}

// Marks free cells along one sensor ray and optionally its occupied endpoint.
void OccupancyMapper::markRay(
  int start_x,
  int start_y,
  int end_x,
  int end_y,
  bool occupied_endpoint)
{
  int x = start_x;
  int y = start_y;
  const int delta_x = std::abs(end_x - start_x);
  const int delta_y = std::abs(end_y - start_y);
  const int step_x = start_x < end_x ? 1 : -1;
  const int step_y = start_y < end_y ? 1 : -1;
  int error = delta_x - delta_y;

  while (true)
  {
    const std::size_t index =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(map_width_cells_) +
      static_cast<std::size_t>(x);

    if (x == end_x && y == end_y)
    {
      if (occupied_endpoint)
      {
        frame_occupied_cells_[index] = 1;
      }
      else
      {
        frame_free_cells_[index] = 1;
      }
      break;
    }

    frame_free_cells_[index] = 1;
    const int doubled_error = 2 * error;
    if (doubled_error > -delta_y)
    {
      error -= delta_y;
      x += step_x;
    }
    if (doubled_error < delta_x)
    {
      error += delta_x;
      y += step_y;
    }
  }
}

// Applies one evidence update per touched cell for the current depth frame.
void OccupancyMapper::applyFrameUpdates()
{
  for (std::size_t index = 0; index < log_odds_grid_.size(); ++index)
  {
    if (frame_occupied_cells_[index] != 0)
    {
      log_odds_grid_[index] = static_cast<int16_t>(std::min(
          max_log_odds_,
          static_cast<int>(log_odds_grid_[index]) + hit_log_odds_));
      observed_grid_[index] = 1;
    }
    else if (frame_free_cells_[index] != 0)
    {
      log_odds_grid_[index] = static_cast<int16_t>(std::max(
          min_log_odds_,
          static_cast<int>(log_odds_grid_[index]) - miss_log_odds_));
      observed_grid_[index] = 1;
    }
  }
}

// Publishes the current occupancy grid and metadata.
void OccupancyMapper::publishMap(const builtin_interfaces::msg::Time &stamp)
{
  nav_msgs::msg::OccupancyGrid map;
  map.header.stamp = stamp;
  map.header.frame_id = map_frame_;
  map.info.map_load_time = map_load_time_;
  map.info.resolution = static_cast<float>(resolution_);
  map.info.width = static_cast<uint32_t>(map_width_cells_);
  map.info.height = static_cast<uint32_t>(map_height_cells_);
  map.info.origin.position.x = -map_width_meters_ * 0.5;
  map.info.origin.position.y = -map_height_meters_ * 0.5;
  map.info.origin.position.z = 0.0;
  map.info.origin.orientation.w = 1.0;
  map.data.resize(log_odds_grid_.size(), -1);

  for (std::size_t index = 0; index < log_odds_grid_.size(); ++index)
  {
    if (observed_grid_[index] == 0)
    {
      map.data[index] = -1;
    }
    else if (log_odds_grid_[index] >= occupied_log_odds_threshold_)
    {
      map.data[index] = 100;
    }
    else if (log_odds_grid_[index] <= free_log_odds_threshold_)
    {
      map.data[index] = 0;
    }
    else
    {
      map.data[index] = -1;
    }
  }

  map_publisher_->publish(map);
  metadata_publisher_->publish(map.info);
}

// Publishes the fixed gravity-aligned transform from orb_map to nav_map.
void OccupancyMapper::publishNavigationTransform(
  const builtin_interfaces::msg::Time &stamp)
{
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = stamp;
  transform.header.frame_id = world_frame_;
  transform.child_frame_id = map_frame_;
  transform.transform.translation.x = navigation_origin_world_x_;
  transform.transform.translation.y = navigation_origin_world_y_;
  transform.transform.translation.z = navigation_origin_world_z_;
  transform.transform.rotation.z = std::sin(navigation_yaw_ * 0.5);
  transform.transform.rotation.w = std::cos(navigation_yaw_ * 0.5);
  transform_broadcaster_->sendTransform(transform);
}

// Clears occupancy evidence while preserving the current navigation frame.
void OccupancyMapper::clearGrid(const char *reason)
{
  const std::size_t observed_cells = static_cast<std::size_t>(
    std::count(observed_grid_.begin(), observed_grid_.end(), static_cast<uint8_t>(1)));
  if (observed_cells > 0)
  {
    RCLCPP_WARN(
      get_logger(), "Clearing occupancy grid (%zu observed cells): %s",
      observed_cells, reason);
  }

  std::fill(log_odds_grid_.begin(), log_odds_grid_.end(), 0);
  std::fill(observed_grid_.begin(), observed_grid_.end(), 0);
  std::fill(frame_free_cells_.begin(), frame_free_cells_.end(), 0);
  std::fill(frame_occupied_cells_.begin(), frame_occupied_cells_.end(), 0);
  processed_frames_ = 0;

  if (navigation_frame_initialized_)
  {
    publishMap(last_message_stamp_);
  }
}

// Clears occupancy evidence and waits for a new navigation-frame origin.
void OccupancyMapper::resetMapper(const char *reason)
{
  clearGrid(reason);
  navigation_frame_initialized_ = false;
  have_last_pose_ = false;
}

// Handles a manual request to clear the current grid.
void OccupancyMapper::clearServiceCallback(
  const std_srvs::srv::Empty::Request::SharedPtr,
  std_srvs::srv::Empty::Response::SharedPtr)
{
  clearGrid("manual request");
}

// Prints compact processing and map-occupancy statistics.
void OccupancyMapper::logStatistics()
{
  const std::size_t observed_cells = static_cast<std::size_t>(
    std::count(observed_grid_.begin(), observed_grid_.end(), static_cast<uint8_t>(1)));
  const std::size_t occupied_cells = static_cast<std::size_t>(std::count_if(
      log_odds_grid_.begin(),
      log_odds_grid_.end(),
      [this](int16_t value)
      {
        return value >= occupied_log_odds_threshold_;
      }));

  RCLCPP_INFO(
    get_logger(),
    "Occupancy stats: synced=%lu processed=%lu valid_depth=%lu obstacles=%lu "
    "floor=%lu observed_cells=%zu occupied_cells=%zu tracking=%s imu=%s",
    static_cast<unsigned long>(synchronized_frames_),
    static_cast<unsigned long>(processed_frames_),
    static_cast<unsigned long>(valid_depth_samples_),
    static_cast<unsigned long>(obstacle_samples_),
    static_cast<unsigned long>(floor_samples_),
    observed_cells,
    occupied_cells,
    tracking_ok_ ? "OK" : "NOT_OK",
    use_imu_ ? (imu_initialized_ ? "READY" : "WAITING") : "DISABLED");
}

// Starts the ROS 2 executor for the occupancy mapper.
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OccupancyMapper>());
  rclcpp::shutdown();
  return 0;
}
