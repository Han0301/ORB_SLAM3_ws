#ifndef YOLO_DET__YOLO_NODE_H_
#define YOLO_DET__YOLO_NODE_H_

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include "perception_interfaces/msg/detection2_d_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "detection_types.h"
#include "tensorrt_detector.h"

namespace yolo_det
{

class YoloNode : public rclcpp::Node
{
public:
  YoloNode();
  ~YoloNode() override;

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & image_message);
  void inference_loop();
  bool image_expired(
    const sensor_msgs::msg::Image::ConstSharedPtr & image_message) const;
  cv::Mat colorBgr8(const sensor_msgs::msg::Image::ConstSharedPtr & message);
  std::vector<Detection2D> process_image(const cv::Mat & image_bgr);
  void publish_detections(
    const std::vector<Detection2D> & detections,
    const sensor_msgs::msg::Image::ConstSharedPtr & source_message);
  void publish_annotated_image(
    const cv::Mat & image_bgr,
    const std::vector<Detection2D> & detections,
    const sensor_msgs::msg::Image::ConstSharedPtr & source_message);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  std::string image_topic_ = "/camera/color/image_raw";
  std::string detections_topic_ = "/yolo/detections_2d";
  std::string annotated_image_topic_ = "/yolo/debug_image";
  std::string annotation_font_path_ =
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc";
  int annotation_font_face_index_{2};
  bool publish_annotated_image_enabled_{false};
  std::string engine_path_;
  double inference_rate_hz_{0.0};
  double max_frame_age_ms_{100.0};
  double confidence_threshold_{0.25};
  double nms_iou_threshold_{0.45};
  std::unique_ptr<TensorRtDetector> detector_;
  rclcpp::Publisher<perception_interfaces::msg::Detection2DArray>::SharedPtr
    detections_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_image_publisher_;

  std::mutex frame_mutex_;
  std::condition_variable frame_condition_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_image_;
  std::thread inference_thread_;
  bool stop_requested_{false};
  std::uint64_t received_frames_{0};
  std::uint64_t replaced_frames_{0};
  std::uint64_t expired_frames_{0};
  std::uint64_t processed_frames_{0};
};

}  // namespace yolo_det

#endif  // YOLO_DET__YOLO_NODE_H_
