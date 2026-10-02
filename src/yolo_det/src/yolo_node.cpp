#include "yolo_node.h"

namespace yolo_det
{

YoloNode::YoloNode()
: Node("yolo_node")
{
  image_topic_ = declare_parameter<std::string>("image_topic", image_topic_);
  detections_topic_ = declare_parameter<std::string>(
    "detections_topic", detections_topic_);
  engine_path_ = declare_parameter<std::string>("engine_path", "");
  inference_rate_hz_ = declare_parameter<double>(
    "inference_rate_hz", inference_rate_hz_);
  max_frame_age_ms_ = declare_parameter<double>(
    "max_frame_age_ms", max_frame_age_ms_);
  confidence_threshold_ = declare_parameter<double>(
    "confidence_threshold", confidence_threshold_);
  nms_iou_threshold_ = declare_parameter<double>(
    "nms_iou_threshold", nms_iou_threshold_);
  annotated_image_topic_ = declare_parameter<std::string>(
    "annotated_image_topic", annotated_image_topic_);
  publish_annotated_image_enabled_ = declare_parameter<bool>(
    "publish_annotated_image", publish_annotated_image_enabled_);
  annotation_font_path_ = declare_parameter<std::string>(
    "annotation_font_path", annotation_font_path_);
  annotation_font_face_index_ = declare_parameter<int>(
    "annotation_font_face_index", annotation_font_face_index_);

  if (confidence_threshold_ < 0.0 || confidence_threshold_ > 1.0 ||
    nms_iou_threshold_ < 0.0 || nms_iou_threshold_ > 1.0)
  {
    throw std::invalid_argument("YOLO confidence and NMS thresholds must be in [0, 1]");
  }
  if (inference_rate_hz_ < 0.0)
  {
    throw std::invalid_argument("inference_rate_hz must be greater than or equal to zero");
  }
  if (max_frame_age_ms_ < 0.0)
  {
    throw std::invalid_argument("max_frame_age_ms must be greater than or equal to zero");
  }
  if (engine_path_.empty())
  {
    engine_path_ = ament_index_cpp::get_package_share_directory("yolo_det") +
      "/models/dfine_s_obj2coco.engine";
  }

  detector_ = std::make_unique<TensorRtDetector>(
    engine_path_, static_cast<float>(confidence_threshold_),
    static_cast<float>(nms_iou_threshold_));
  detections_publisher_ =
    create_publisher<perception_interfaces::msg::Detection2DArray>(
    detections_topic_, rclcpp::SensorDataQoS());
  image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_,
    rclcpp::SensorDataQoS(),
    std::bind(&YoloNode::image_callback, this, std::placeholders::_1));
  inference_thread_ = std::thread(&YoloNode::inference_loop, this);

  if (inference_rate_hz_ > 0.0)
  {
    RCLCPP_INFO(
      get_logger(),
      "%s TensorRT ready: input '%s', detections '%s', maximum %.1f Hz",
      detector_->backend_name(), image_topic_.c_str(), detections_topic_.c_str(),
      inference_rate_hz_);
  }
  else
  {
    RCLCPP_INFO(
      get_logger(),
      "%s TensorRT ready: input '%s', detections '%s', rate limited by inference",
      detector_->backend_name(), image_topic_.c_str(), detections_topic_.c_str());
  }
}

YoloNode::~YoloNode()
{
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    stop_requested_ = true;
    latest_image_.reset();
  }
  frame_condition_.notify_one();
  if (inference_thread_.joinable())
  {
    inference_thread_.join();
  }
}

void YoloNode::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_message)
{
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    ++received_frames_;
    if (latest_image_)
    {
      ++replaced_frames_;
    }
    latest_image_ = image_message;
  }
  frame_condition_.notify_one();
}

void YoloNode::inference_loop()
{
  using SteadyClock = std::chrono::steady_clock;
  const SteadyClock::duration minimum_interval = inference_rate_hz_ > 0.0 ?
    std::chrono::duration_cast<SteadyClock::duration>(
    std::chrono::duration<double>(1.0 / inference_rate_hz_)) :
    SteadyClock::duration::zero();
  SteadyClock::time_point next_inference_time = SteadyClock::time_point::min();

  while (true)
  {
    sensor_msgs::msg::Image::ConstSharedPtr image_message;
    std::uint64_t received_frames = 0;
    std::uint64_t replaced_frames = 0;
    {
      std::unique_lock<std::mutex> lock(frame_mutex_);
      frame_condition_.wait(
        lock,
        [this]()
        {
          return stop_requested_ || latest_image_;
        });
      if (stop_requested_)
      {
        return;
      }

      if (inference_rate_hz_ > 0.0 && SteadyClock::now() < next_inference_time)
      {
        frame_condition_.wait_until(
          lock,
          next_inference_time,
          [this]()
          {
            return stop_requested_;
          });
        if (stop_requested_)
        {
          return;
        }
      }

      image_message = std::move(latest_image_);
      latest_image_.reset();
      received_frames = received_frames_;
      replaced_frames = replaced_frames_;
    }

    if (image_expired(image_message))
    {
      {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        ++expired_frames_;
      }
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Discarding an image older than %.1f ms",
        max_frame_age_ms_);
      continue;
    }

    const SteadyClock::time_point frame_processing_start = SteadyClock::now();
    if (inference_rate_hz_ > 0.0)
    {
      next_inference_time = frame_processing_start + minimum_interval;
    }

    try
    {
      const cv::Mat image_bgr = colorBgr8(image_message);
      const SteadyClock::time_point inference_start = SteadyClock::now();
      const std::vector<Detection2D> detections = process_image(image_bgr);
      const float inference_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(
          SteadyClock::now() - inference_start).count());

      publish_detections(detections, image_message);
      if (publish_annotated_image_enabled_)
      {
        publish_annotated_image(image_bgr, detections, image_message);
      }

      ++processed_frames_;
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "%s detected %zu objects in %.2f ms; frames received=%lu processed=%lu replaced=%lu",
        detector_->backend_name(), detections.size(), inference_ms,
        static_cast<unsigned long>(received_frames),
        static_cast<unsigned long>(processed_frames_),
        static_cast<unsigned long>(replaced_frames));
      for (const Detection2D & detection : detections)
      {
        RCLCPP_DEBUG(
          get_logger(), "class=%d confidence=%.3f box=[%.1f, %.1f, %.1f, %.1f]",
          detection.class_id, detection.confidence,
          detection.box.x_min, detection.box.y_min,
          detection.box.x_max, detection.box.y_max);
      }
    }
    catch (const std::exception & error)
    {
      RCLCPP_ERROR(get_logger(), "Failed to process image: %s", error.what());
    }
  }
}

bool YoloNode::image_expired(
  const sensor_msgs::msg::Image::ConstSharedPtr & image_message) const
{
  if (max_frame_age_ms_ <= 0.0 ||
    (image_message->header.stamp.sec == 0 && image_message->header.stamp.nanosec == 0))
  {
    return false;
  }

  const rclcpp::Time image_stamp(
    image_message->header.stamp,
    get_clock()->get_clock_type());
  const std::int64_t age_nanoseconds = (get_clock()->now() - image_stamp).nanoseconds();
  return age_nanoseconds > static_cast<std::int64_t>(max_frame_age_ms_ * 1.0e6);
}

cv::Mat YoloNode::colorBgr8(
  const sensor_msgs::msg::Image::ConstSharedPtr & message)
{
  return cv_bridge::toCvCopy(
    message, sensor_msgs::image_encodings::BGR8)->image;
}

std::vector<Detection2D> YoloNode::process_image(const cv::Mat & image_bgr)
{
  return detector_->detect(image_bgr);
}

void YoloNode::publish_detections(
  const std::vector<Detection2D> & detections,
  const sensor_msgs::msg::Image::ConstSharedPtr & source_message)
{
  perception_interfaces::msg::Detection2DArray message;
  message.header = source_message->header;
  message.detections.reserve(detections.size());

  for (const Detection2D & detection : detections)
  {
    perception_interfaces::msg::Detection2D output;
    output.box.x_min = detection.box.x_min;
    output.box.y_min = detection.box.y_min;
    output.box.x_max = detection.box.x_max;
    output.box.y_max = detection.box.y_max;
    output.class_id = detection.class_id;
    output.confidence = detection.confidence;
    message.detections.push_back(std::move(output));
  }

  detections_publisher_->publish(message);
}

void YoloNode::publish_annotated_image(
  const cv::Mat & image_bgr,
  const std::vector<Detection2D> & detections,
  const sensor_msgs::msg::Image::ConstSharedPtr & source_message)
{
  if (!annotated_image_publisher_)
  {
    annotated_image_publisher_ = create_publisher<sensor_msgs::msg::Image>(
      annotated_image_topic_, rclcpp::SensorDataQoS());
    RCLCPP_INFO(
      get_logger(), "Publishing annotated images on '%s'.",
      annotated_image_topic_.c_str());
  }

  cv::Mat annotated = detector_->draw_detections(
    image_bgr, detections, annotation_font_path_, annotation_font_face_index_);
  auto output_message = cv_bridge::CvImage(
    source_message->header,
    sensor_msgs::image_encodings::BGR8,
    std::move(annotated)).toImageMsg();
  annotated_image_publisher_->publish(*output_message);
}

}  // namespace yolo_det

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_det::YoloNode>());
  rclcpp::shutdown();
  return 0;
}
