#ifndef MULTI_OBJECT_TRACKER__TRACKER_NODE_H_
#define MULTI_OBJECT_TRACKER__TRACKER_NODE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "perception_interfaces/msg/detection2_d_array.hpp"
#include "perception_interfaces/msg/tracked_object2_d.hpp"
#include "perception_interfaces/msg/tracked_object2_d_array.hpp"
#include "rclcpp/rclcpp.hpp"

#include "byte_tracker.h"

namespace multi_object_tracker
{

class TrackerNode : public rclcpp::Node
{
public:
  TrackerNode();

private:
  void detections_callback(
    const perception_interfaces::msg::Detection2DArray::ConstSharedPtr & message);
  TrackerConfig load_config();

  std::string detections_topic_{"/yolo/detections_2d"};
  std::string tracks_topic_{"/tracking/tracks_2d"};
  std::unique_ptr<ByteTracker> tracker_;
  rclcpp::Subscription<perception_interfaces::msg::Detection2DArray>::SharedPtr
    detections_subscription_;
  rclcpp::Publisher<perception_interfaces::msg::TrackedObject2DArray>::SharedPtr
    tracks_publisher_;
  std::uint64_t input_message_count_{0};
  std::uint64_t output_track_count_{0};
};

}  // namespace multi_object_tracker

#endif  // MULTI_OBJECT_TRACKER__TRACKER_NODE_H_
