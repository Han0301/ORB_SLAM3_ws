#include "tensorrt_detector.h"

namespace yolo_det
{
// 私有辅助函数和常量的命名空间
namespace
{

// 当前 yolov8s.engine 的 COCO 80 类，顺序与模型 class_id 一致。
constexpr std::array<const char *, 80> kCoco80ClassNamesZh = {
  "人", "自行车", "汽车", "摩托车", "飞机",
  "公交车", "火车", "卡车", "船", "交通信号灯",
  "消防栓", "停止标志", "停车计时器", "长椅", "鸟",
  "猫", "狗", "马", "羊", "牛",
  "大象", "熊", "斑马", "长颈鹿", "背包",
  "雨伞", "手提包", "领带", "行李箱", "飞盘",
  "双板滑雪板", "单板滑雪板", "运动球", "风筝", "棒球棒",
  "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
  "高脚杯", "杯子", "叉子", "刀", "勺子",
  "碗", "香蕉", "苹果", "三明治", "橙子",
  "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈",
  "蛋糕", "椅子", "沙发", "盆栽", "床",
  "餐桌", "马桶", "电视", "笔记本电脑", "鼠标",
  "遥控器", "键盘", "手机", "微波炉", "烤箱",
  "烤面包机", "水槽", "冰箱", "书", "时钟",
  "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷",
};

cv::Scalar detection_color(std::size_t index)
{
  // 按框序号分散色相；同类的多个框也使用不同颜色。
  const double hue = std::fmod(static_cast<double>(index) * 137.507764, 360.0);
  const cv::Mat hsv(1, 1, CV_32FC3, cv::Scalar(hue, 0.70, 1.0));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3f color = bgr.at<cv::Vec3f>(0, 0);
  return cv::Scalar(color[0] * 255.0F, color[1] * 255.0F, color[2] * 255.0F);
}

// 检查 CUDA 操作是否成功
void check_cuda(cudaError_t result, const char * operation)
{
  if (result != cudaSuccess)
  {
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

// 读取 TensorRT 引擎文件
std::vector<char> read_engine(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
  {
    throw std::runtime_error("Could not open TensorRT engine: " + path);
  }

  const auto file_size = file.tellg();
  if (file_size <= static_cast<std::streamoff>(sizeof(int32_t)))
  {
    throw std::runtime_error("TensorRT engine file is empty or truncated: " + path);
  }

  file.seekg(0, std::ios::beg);
  std::vector<char> bytes(static_cast<std::size_t>(file_size));
  file.read(bytes.data(), file_size);
  if (!file)
  {
    throw std::runtime_error("Could not read TensorRT engine: " + path);
  }
  return bytes;
}

// 计算两个边界框的交并比（IoU）
float cal_iou(const BoundingBox2D & lhs, const BoundingBox2D & rhs)
{
  const float left = std::max(lhs.x_min, rhs.x_min);
  const float top = std::max(lhs.y_min, rhs.y_min);
  const float right = std::min(lhs.x_max, rhs.x_max);
  const float bottom = std::min(lhs.y_max, rhs.y_max);
  const float intersection = std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  const float lhs_area = std::max(0.0F, lhs.x_max - lhs.x_min) *
    std::max(0.0F, lhs.y_max - lhs.y_min);
  const float rhs_area = std::max(0.0F, rhs.x_max - rhs.x_min) *
    std::max(0.0F, rhs.y_max - rhs.y_min);
  const float union_area = lhs_area + rhs_area - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

}  // namespace

void TensorRtLogger::log(Severity severity, const char * message) noexcept
{
  if (severity <= Severity::kWARNING)
  {
    std::cerr << "[TensorRT] " << message << '\n';
  }
}

TensorRtDetector::TensorRtDetector
(
  const std::string & engine_path,
  float confidence_threshold,
  float nms_iou_threshold)
: confidence_threshold_(confidence_threshold),
  nms_iou_threshold_(nms_iou_threshold)
{
  const std::vector<char> file_bytes = read_engine(engine_path);
  const char * engine_data = file_bytes.data();
  std::size_t engine_size = file_bytes.size();

  int32_t metadata_size = 0;
  std::memcpy(&metadata_size, file_bytes.data(), sizeof(metadata_size));
  const std::size_t metadata_offset = sizeof(metadata_size);
  const bool has_ultralytics_metadata = metadata_size > 0 &&
    metadata_offset + static_cast<std::size_t>(metadata_size) < file_bytes.size() &&
    file_bytes[metadata_offset] == '{';
  if (has_ultralytics_metadata)
  {
    const std::size_t engine_offset = metadata_offset +
      static_cast<std::size_t>(metadata_size);
    engine_data = file_bytes.data() + engine_offset;
    engine_size = file_bytes.size() - engine_offset;
  }

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_)
  {
    throw std::runtime_error("Could not create TensorRT runtime");
  }
  engine_.reset(runtime_->deserializeCudaEngine(engine_data, engine_size));
  if (!engine_)
  {
    throw std::runtime_error("Could not deserialize TensorRT engine");
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_)
  {
    throw std::runtime_error("Could not create TensorRT execution context");
  }

  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  for (int index = 0; index < engine_->getNbIOTensors(); ++index)
  {
    const char * name = engine_->getIOTensorName(index);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
    {
      input_names.emplace_back(name);
    }
    else
    {
      output_names.emplace_back(name);
    }
  }

  check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
  if (input_names.size() == 2U && output_names.size() == 3U &&
    std::find(input_names.begin(), input_names.end(), "images") != input_names.end() &&
    std::find(input_names.begin(), input_names.end(), "orig_target_sizes") != input_names.end() &&
    std::find(output_names.begin(), output_names.end(), "labels") != output_names.end() &&
    std::find(output_names.begin(), output_names.end(), "boxes") != output_names.end() &&
    std::find(output_names.begin(), output_names.end(), "scores") != output_names.end())
  {
    backend_ = Backend::kDFine;
    configure_dfine_engine();
  }
  else if (input_names.size() == 1U && output_names.size() == 1U)
  {
    backend_ = Backend::kYolo;
    input_name_ = input_names.front();
    output_name_ = output_names.front();
    configure_yolo_engine();
  }
  else
  {
    throw std::runtime_error(
      "Unsupported TensorRT detector interface: expected YOLO or D-FINE tensors");
  }
}

void TensorRtDetector::configure_yolo_engine()
{
  if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
    engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT)
  {
    throw std::runtime_error("Expected FP32 YOLO TensorRT input/output tensors");
  }
  const nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
  const nvinfer1::Dims output_dims = engine_->getTensorShape(output_name_.c_str());
  if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
    output_dims.nbDims != 3 || output_dims.d[0] != 1 || output_dims.d[1] <= 4)
  {
    throw std::runtime_error("Unexpected YOLOv8 TensorRT tensor dimensions");
  }

  input_height_ = input_dims.d[2];
  input_width_ = input_dims.d[3];
  class_count_ = output_dims.d[1] - 4;
  candidate_count_ = output_dims.d[2];
  input_buffer_.resize(3U * input_height_ * input_width_);
  output_buffer_.resize(static_cast<std::size_t>(output_dims.d[1]) * candidate_count_);

  check_cuda(
    cudaMalloc(&device_image_input_, input_buffer_.size() * sizeof(float)),
    "cudaMalloc(input)");
  check_cuda(
    cudaMalloc(&device_yolo_output_, output_buffer_.size() * sizeof(float)),
    "cudaMalloc(output)");

  if (!context_->setTensorAddress(input_name_.c_str(), device_image_input_) ||
    !context_->setTensorAddress(output_name_.c_str(), device_yolo_output_))
  {
    throw std::runtime_error("Could not bind YOLO TensorRT input/output tensors");
  }
}

void TensorRtDetector::configure_dfine_engine()
{
  input_name_ = "images";
  if (engine_->getTensorDataType("images") != nvinfer1::DataType::kFLOAT ||
    engine_->getTensorDataType("orig_target_sizes") != nvinfer1::DataType::kINT64 ||
    engine_->getTensorDataType("labels") != nvinfer1::DataType::kINT64 ||
    engine_->getTensorDataType("boxes") != nvinfer1::DataType::kFLOAT ||
    engine_->getTensorDataType("scores") != nvinfer1::DataType::kFLOAT)
  {
    throw std::runtime_error("Unexpected D-FINE TensorRT tensor data types");
  }

  const nvinfer1::Dims image_dims = engine_->getTensorShape("images");
  const nvinfer1::Dims size_dims = engine_->getTensorShape("orig_target_sizes");
  const nvinfer1::Dims label_dims = engine_->getTensorShape("labels");
  const nvinfer1::Dims box_dims = engine_->getTensorShape("boxes");
  const nvinfer1::Dims score_dims = engine_->getTensorShape("scores");
  if (image_dims.nbDims != 4 || image_dims.d[0] != 1 || image_dims.d[1] != 3 ||
    size_dims.nbDims != 2 || size_dims.d[0] != 1 || size_dims.d[1] != 2 ||
    label_dims.nbDims != 2 || label_dims.d[0] != 1 || label_dims.d[1] <= 0 ||
    box_dims.nbDims != 3 || box_dims.d[0] != 1 ||
    box_dims.d[1] != label_dims.d[1] || box_dims.d[2] != 4 ||
    score_dims.nbDims != 2 || score_dims.d[0] != 1 ||
    score_dims.d[1] != label_dims.d[1])
  {
    throw std::runtime_error("Unexpected D-FINE TensorRT tensor dimensions");
  }

  input_height_ = image_dims.d[2];
  input_width_ = image_dims.d[3];
  candidate_count_ = label_dims.d[1];
  input_buffer_.resize(3U * input_height_ * input_width_);
  labels_buffer_.resize(candidate_count_);
  boxes_buffer_.resize(static_cast<std::size_t>(candidate_count_) * 4U);
  scores_buffer_.resize(candidate_count_);

  check_cuda(
    cudaMalloc(&device_image_input_, input_buffer_.size() * sizeof(float)),
    "cudaMalloc(images)");
  check_cuda(
    cudaMalloc(&device_orig_target_sizes_, orig_target_sizes_.size() * sizeof(std::int64_t)),
    "cudaMalloc(orig_target_sizes)");
  check_cuda(
    cudaMalloc(&device_labels_, labels_buffer_.size() * sizeof(std::int64_t)),
    "cudaMalloc(labels)");
  check_cuda(
    cudaMalloc(&device_boxes_, boxes_buffer_.size() * sizeof(float)),
    "cudaMalloc(boxes)");
  check_cuda(
    cudaMalloc(&device_scores_, scores_buffer_.size() * sizeof(float)),
    "cudaMalloc(scores)");

  if (!context_->setTensorAddress("images", device_image_input_) ||
    !context_->setTensorAddress("orig_target_sizes", device_orig_target_sizes_) ||
    !context_->setTensorAddress("labels", device_labels_) ||
    !context_->setTensorAddress("boxes", device_boxes_) ||
    !context_->setTensorAddress("scores", device_scores_))
  {
    throw std::runtime_error("Could not bind D-FINE TensorRT input/output tensors");
  }
}

TensorRtDetector::~TensorRtDetector()
{
  if (device_scores_ != nullptr)
  {
    cudaFree(device_scores_);
  }
  if (device_boxes_ != nullptr)
  {
    cudaFree(device_boxes_);
  }
  if (device_labels_ != nullptr)
  {
    cudaFree(device_labels_);
  }
  if (device_orig_target_sizes_ != nullptr)
  {
    cudaFree(device_orig_target_sizes_);
  }
  if (device_yolo_output_ != nullptr)
  {
    cudaFree(device_yolo_output_);
  }
  if (device_image_input_ != nullptr)
  {
    cudaFree(device_image_input_);
  }
  if (stream_ != nullptr)
  {
    cudaStreamDestroy(stream_);
  }
}

const char * TensorRtDetector::backend_name() const noexcept
{
  return backend_ == Backend::kDFine ? "D-FINE-S" : "YOLO";
}

TensorRtDetector::LetterboxInfo TensorRtDetector::preprocess(const cv::Mat & image_bgr)
{
  LetterboxInfo letterbox{1.0F, 0.0F, 0.0F};
  cv::Mat network_image;
  if (backend_ == Backend::kYolo)
  {
    letterbox.scale = std::min(
      static_cast<float>(input_width_) / image_bgr.cols,
      static_cast<float>(input_height_) / image_bgr.rows);
    const int resized_width = static_cast<int>(std::round(image_bgr.cols * letterbox.scale));
    const int resized_height = static_cast<int>(std::round(image_bgr.rows * letterbox.scale));
    const int horizontal_padding = input_width_ - resized_width;
    const int vertical_padding = input_height_ - resized_height;
    const int left = horizontal_padding / 2;
    const int top = vertical_padding / 2;

    cv::Mat resized;
    cv::resize(image_bgr, resized, cv::Size(resized_width, resized_height));
    cv::copyMakeBorder(
      resized, network_image, top, vertical_padding - top,
      left, horizontal_padding - left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    letterbox.pad_x = static_cast<float>(left);
    letterbox.pad_y = static_cast<float>(top);
  }
  else
  {
    cv::resize(image_bgr, network_image, cv::Size(input_width_, input_height_));
  }

  cv::cvtColor(network_image, network_image, cv::COLOR_BGR2RGB);
  network_image.convertTo(network_image, CV_32FC3, 1.0 / 255.0);

  const std::size_t channel_size = static_cast<std::size_t>(input_height_) * input_width_;
  std::vector<cv::Mat> channels;
  channels.reserve(3);
  for (int channel = 0; channel < 3; ++channel)
  {
    channels.emplace_back(
      input_height_, input_width_, CV_32FC1,
      input_buffer_.data() + channel * channel_size);
  }
  cv::split(network_image, channels);
  return letterbox;
}

std::vector<Detection2D> TensorRtDetector::decode_yolo(
  const cv::Size & original_size,
  const LetterboxInfo & letterbox) const
{
  std::vector<Detection2D> candidates;
  candidates.reserve(candidate_count_);

  const auto output_value = [this](int channel, int candidate) {
      return output_buffer_[static_cast<std::size_t>(channel) * candidate_count_ + candidate];
    };

  for (int candidate = 0; candidate < candidate_count_; ++candidate)
  {
    int class_id = 0;
    float confidence = -std::numeric_limits<float>::infinity();
    for (int class_index = 0; class_index < class_count_; ++class_index)
    {
      const float score = output_value(class_index + 4, candidate);
      if (score > confidence)
      {
        confidence = score;
        class_id = class_index;
      }
    }
    if (confidence < confidence_threshold_)
    {
      continue;
    }

    const float center_x = output_value(0, candidate);
    const float center_y = output_value(1, candidate);
    const float width = output_value(2, candidate);
    const float height = output_value(3, candidate);
    BoundingBox2D box{
      (center_x - width * 0.5F - letterbox.pad_x) / letterbox.scale,
      (center_y - height * 0.5F - letterbox.pad_y) / letterbox.scale,
      (center_x + width * 0.5F - letterbox.pad_x) / letterbox.scale,
      (center_y + height * 0.5F - letterbox.pad_y) / letterbox.scale};
    box.x_min = std::clamp(box.x_min, 0.0F, static_cast<float>(original_size.width - 1));
    box.y_min = std::clamp(box.y_min, 0.0F, static_cast<float>(original_size.height - 1));
    box.x_max = std::clamp(box.x_max, 0.0F, static_cast<float>(original_size.width - 1));
    box.y_max = std::clamp(box.y_max, 0.0F, static_cast<float>(original_size.height - 1));
    candidates.push_back(Detection2D{box, class_id, confidence});
  }

  std::sort(
    candidates.begin(), candidates.end(),
    [](const Detection2D & lhs, const Detection2D & rhs) {
      return lhs.confidence > rhs.confidence;
    });

  std::vector<Detection2D> detections;
  detections.reserve(candidates.size());
  for (const Detection2D & candidate : candidates)
  {
    bool suppressed = false;
    for (const Detection2D & selected : detections)
    {
      if (candidate.class_id == selected.class_id &&
        cal_iou(candidate.box, selected.box) > nms_iou_threshold_)
      {
        suppressed = true;
        break;
      }
    }
    if (!suppressed)
    {
      detections.push_back(candidate);
    }
  }
  return detections;
}

std::vector<Detection2D> TensorRtDetector::decode_dfine(
  const cv::Size & original_size) const
{
  std::vector<Detection2D> detections;
  detections.reserve(candidate_count_);

  for (int candidate = 0; candidate < candidate_count_; ++candidate)
  {
    const float confidence = scores_buffer_[static_cast<std::size_t>(candidate)];
    if (!std::isfinite(confidence) || confidence < confidence_threshold_)
    {
      continue;
    }

    const std::size_t box_offset = static_cast<std::size_t>(candidate) * 4U;
    BoundingBox2D box{
      boxes_buffer_[box_offset],
      boxes_buffer_[box_offset + 1U],
      boxes_buffer_[box_offset + 2U],
      boxes_buffer_[box_offset + 3U]};
    box.x_min = std::clamp(box.x_min, 0.0F, static_cast<float>(original_size.width - 1));
    box.y_min = std::clamp(box.y_min, 0.0F, static_cast<float>(original_size.height - 1));
    box.x_max = std::clamp(box.x_max, 0.0F, static_cast<float>(original_size.width - 1));
    box.y_max = std::clamp(box.y_max, 0.0F, static_cast<float>(original_size.height - 1));
    if (box.x_max <= box.x_min || box.y_max <= box.y_min)
    {
      continue;
    }

    const std::int64_t class_id = labels_buffer_[static_cast<std::size_t>(candidate)];
    if (class_id < 0 || class_id > std::numeric_limits<std::int32_t>::max())
    {
      continue;
    }
    detections.push_back(
      Detection2D{box, static_cast<std::int32_t>(class_id), confidence});
  }
  return detections;
}

std::vector<Detection2D> TensorRtDetector::detect(const cv::Mat & image_bgr)
{
  if (image_bgr.empty() || image_bgr.type() != CV_8UC3)
  {
    throw std::invalid_argument("TensorRtDetector expects a non-empty CV_8UC3 BGR image");
  }

  const LetterboxInfo letterbox = preprocess(image_bgr);
  check_cuda(
    cudaMemcpyAsync(
      device_image_input_, input_buffer_.data(), input_buffer_.size() * sizeof(float),
      cudaMemcpyHostToDevice, stream_),
    "cudaMemcpyAsync(images)");
  if (backend_ == Backend::kDFine)
  {
    orig_target_sizes_[0] = image_bgr.cols;
    orig_target_sizes_[1] = image_bgr.rows;
    check_cuda(
      cudaMemcpyAsync(
        device_orig_target_sizes_, orig_target_sizes_.data(),
        orig_target_sizes_.size() * sizeof(std::int64_t),
        cudaMemcpyHostToDevice, stream_),
      "cudaMemcpyAsync(orig_target_sizes)");
  }

  if (!context_->enqueueV3(stream_))
  {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }
  if (backend_ == Backend::kDFine)
  {
    check_cuda(
      cudaMemcpyAsync(
        labels_buffer_.data(), device_labels_, labels_buffer_.size() * sizeof(std::int64_t),
        cudaMemcpyDeviceToHost, stream_),
      "cudaMemcpyAsync(labels)");
    check_cuda(
      cudaMemcpyAsync(
        boxes_buffer_.data(), device_boxes_, boxes_buffer_.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream_),
      "cudaMemcpyAsync(boxes)");
    check_cuda(
      cudaMemcpyAsync(
        scores_buffer_.data(), device_scores_, scores_buffer_.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream_),
      "cudaMemcpyAsync(scores)");
  }
  else
  {
    check_cuda(
      cudaMemcpyAsync(
        output_buffer_.data(), device_yolo_output_, output_buffer_.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream_),
      "cudaMemcpyAsync(output)");
  }
  check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  if (backend_ == Backend::kDFine)
  {
    return decode_dfine(image_bgr.size());
  }
  return decode_yolo(image_bgr.size(), letterbox);
}

cv::Mat TensorRtDetector::draw_detections(
  const cv::Mat & image_bgr,
  const std::vector<Detection2D> & detections,
  const std::string & font_path,
  int font_face_index)
{
  if (image_bgr.empty() || image_bgr.type() != CV_8UC3)
  {
    throw std::invalid_argument("draw_detections expects a non-empty CV_8UC3 BGR image");
  }

  cv::Mat annotated = image_bgr.clone();
  if (detections.empty())
  {
    return annotated;
  }
  if (!annotation_font_ || annotation_font_path_ != font_path ||
    annotation_font_face_index_ != font_face_index)
  {
    auto font = cv::freetype::createFreeType2();
    font->loadFontData(font_path, font_face_index);
    annotation_font_ = std::move(font);
    annotation_font_path_ = font_path;
    annotation_font_face_index_ = font_face_index;
  }

  constexpr int box_thickness = 2;
  constexpr int font_height = 22;
  constexpr int text_thickness = -1;
  constexpr int text_padding = 4;

  for (std::size_t index = 0; index < detections.size(); ++index)
  {
    const Detection2D & detection = detections[index];
    const int x_min = std::clamp(
      static_cast<int>(std::lround(detection.box.x_min)), 0, image_bgr.cols - 1);
    const int y_min = std::clamp(
      static_cast<int>(std::lround(detection.box.y_min)), 0, image_bgr.rows - 1);
    const int x_max = std::clamp(
      static_cast<int>(std::lround(detection.box.x_max)), 0, image_bgr.cols - 1);
    const int y_max = std::clamp(
      static_cast<int>(std::lround(detection.box.y_max)), 0, image_bgr.rows - 1);
    if (x_max <= x_min || y_max <= y_min)
    {
      continue;
    }

    const cv::Scalar box_color = detection_color(index);
    const double luminance =
      0.114 * box_color[0] + 0.587 * box_color[1] + 0.299 * box_color[2];
    const cv::Scalar text_color = luminance > 150.0 ?
      cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
    cv::rectangle(
      annotated, cv::Point(x_min, y_min), cv::Point(x_max, y_max),
      box_color, box_thickness);

    std::ostringstream label_stream;
    const char * class_name = detection.class_id >= 0 &&
      static_cast<std::size_t>(detection.class_id) < kCoco80ClassNamesZh.size() ?
      kCoco80ClassNamesZh[static_cast<std::size_t>(detection.class_id)] : "未知类别";
    label_stream << class_name << " " << std::fixed << std::setprecision(2) <<
      detection.confidence;
    const std::string label = label_stream.str();
    int baseline = 0;
    const cv::Size text_size = annotation_font_->getTextSize(
      label, font_height, text_thickness, &baseline);
    const int label_width = text_size.width + 2 * text_padding;
    const int label_height = text_size.height + baseline + 2 * text_padding;
    const int label_x = std::clamp(x_min, 0, std::max(0, annotated.cols - label_width));
    const int label_y = std::clamp(
      y_min >= label_height ? y_min - label_height : y_min,
      0, std::max(0, annotated.rows - label_height));
    cv::rectangle(
      annotated,
      cv::Point(label_x, label_y),
      cv::Point(
        std::min(annotated.cols - 1, label_x + label_width),
        std::min(annotated.rows - 1, label_y + label_height)),
      box_color, cv::FILLED);
    annotation_font_->putText(
      annotated, label,
      cv::Point(label_x + text_padding, label_y + text_padding + text_size.height),
      font_height, text_color, text_thickness, cv::LINE_AA, true);
  }

  return annotated;
}

}  // namespace yolo_det
