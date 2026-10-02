#include "tracker_node.h"

namespace multi_object_tracker
{

TrackerNode::TrackerNode()
: Node("tracker_node")
{
  detections_topic_ = declare_parameter<std::string>(
    "detections_topic", detections_topic_);
  tracks_topic_ = declare_parameter<std::string>("tracks_topic", tracks_topic_);
  tracker_ = std::make_unique<ByteTracker>(load_config());

  tracks_publisher_ =
    create_publisher<perception_interfaces::msg::TrackedObject2DArray>(
    tracks_topic_, rclcpp::SensorDataQoS());
  detections_subscription_ =
    create_subscription<perception_interfaces::msg::Detection2DArray>(
    detections_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&TrackerNode::detections_callback, this, std::placeholders::_1));

  RCLCPP_INFO(
    get_logger(), "Multi-object tracker ready: input '%s', output '%s'",
    detections_topic_.c_str(), tracks_topic_.c_str());
}

void TrackerNode::detections_callback(
  const perception_interfaces::msg::Detection2DArray::ConstSharedPtr & message)
{
  std::vector<TrackerDetection> detections;
  detections.reserve(message->detections.size());
  for (const auto & input : message->detections)
  {
    detections.push_back(TrackerDetection{
      Box2D{
        input.box.x_min,
        input.box.y_min,
        input.box.x_max,
        input.box.y_max},
      input.class_id,
      input.confidence});
  }

  const double timestamp_sec = rclcpp::Time(message->header.stamp).seconds();
  const std::vector<TrackedObject> tracked_objects = tracker_->update(
    detections, timestamp_sec);

  perception_interfaces::msg::TrackedObject2DArray output;
  output.header = message->header;
  output.tracks.reserve(tracked_objects.size());
  for (const TrackedObject & tracked_object : tracked_objects)
  {
    perception_interfaces::msg::TrackedObject2D object_message;
    object_message.track_id = tracked_object.track_id;
    object_message.box.x_min = static_cast<float>(tracked_object.box.x_min);
    object_message.box.y_min = static_cast<float>(tracked_object.box.y_min);
    object_message.box.x_max = static_cast<float>(tracked_object.box.x_max);
    object_message.box.y_max = static_cast<float>(tracked_object.box.y_max);
    object_message.class_id = tracked_object.class_id;
    object_message.confidence = static_cast<float>(tracked_object.confidence);
    output.tracks.push_back(std::move(object_message));
  }
  tracks_publisher_->publish(output);

  ++input_message_count_;
  output_track_count_ += tracked_objects.size();
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 2000,
    "Tracking: messages=%lu detections=%zu output_tracks=%zu cumulative_tracks=%lu active=%zu",
    static_cast<unsigned long>(input_message_count_),
    detections.size(),
    tracked_objects.size(),
    static_cast<unsigned long>(output_track_count_),
    tracker_->active_track_count());
}

TrackerConfig TrackerNode::load_config()
{
  TrackerConfig config;
  config.high_confidence_threshold = declare_parameter<double>(
    "high_confidence_threshold", config.high_confidence_threshold);
  config.low_confidence_threshold = declare_parameter<double>(
    "low_confidence_threshold", config.low_confidence_threshold);
  config.new_track_threshold = declare_parameter<double>(
    "new_track_threshold", config.new_track_threshold);
  config.high_match_iou_threshold = declare_parameter<double>(
    "high_match_iou_threshold", config.high_match_iou_threshold);
  config.low_match_iou_threshold = declare_parameter<double>(
    "low_match_iou_threshold", config.low_match_iou_threshold);
  config.tentative_match_iou_threshold = declare_parameter<double>(
    "tentative_match_iou_threshold", config.tentative_match_iou_threshold);
  config.confidence_score_fusion_weight = declare_parameter<double>(
    "confidence_score_fusion_weight", config.confidence_score_fusion_weight);
  config.mahalanobis_threshold = declare_parameter<double>(
    "mahalanobis_threshold", config.mahalanobis_threshold);
  const int min_confirmed_hits = declare_parameter<int>(
    "min_confirmed_hits", static_cast<int>(config.min_confirmed_hits));
  if (min_confirmed_hits <= 0)
  {
    throw std::invalid_argument("min_confirmed_hits must be greater than zero");
  }
  config.min_confirmed_hits = static_cast<std::uint32_t>(min_confirmed_hits);
  config.max_lost_time_sec = declare_parameter<double>(
    "max_lost_time_sec", config.max_lost_time_sec);
  config.expected_frame_rate = declare_parameter<double>(
    "expected_frame_rate", config.expected_frame_rate);
  config.max_dt_sec = declare_parameter<double>("max_dt_sec", config.max_dt_sec);
  config.reset_after_gap_sec = declare_parameter<double>(
    "reset_after_gap_sec", config.reset_after_gap_sec);
  config.class_aware_association = declare_parameter<bool>(
    "class_aware_association", config.class_aware_association);
  config.kalman_position_noise_weight = declare_parameter<double>(
    "kalman_position_noise_weight", config.kalman_position_noise_weight);
  config.kalman_velocity_noise_weight = declare_parameter<double>(
    "kalman_velocity_noise_weight", config.kalman_velocity_noise_weight);
  return config;
}

}  // namespace multi_object_tracker

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<multi_object_tracker::TrackerNode>());
  rclcpp::shutdown();
  return 0;
}
