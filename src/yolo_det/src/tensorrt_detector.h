#ifndef YOLO_DET__TENSORRT_DETECTOR_H_
#define YOLO_DET__TENSORRT_DETECTOR_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/imgproc.hpp>

#include "detection_types.h"

namespace yolo_det
{

class TensorRtLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * message) noexcept override;
};

class TensorRtDetector
{
public:
  TensorRtDetector(
    const std::string & engine_path,
    float confidence_threshold,
    float nms_iou_threshold);
  ~TensorRtDetector();

  TensorRtDetector(const TensorRtDetector &) = delete;
  TensorRtDetector & operator=(const TensorRtDetector &) = delete;

  std::vector<Detection2D> detect(const cv::Mat & image_bgr);
  const char * backend_name() const noexcept;
  cv::Mat draw_detections(
    const cv::Mat & image_bgr,
    const std::vector<Detection2D> & detections,
    const std::string & font_path,
    int font_face_index);

private:
  enum class Backend
  {
    kYolo,
    kDFine,
  };

  struct LetterboxInfo
  {
    float scale;
    float pad_x;
    float pad_y;
  };

  void configure_yolo_engine();
  void configure_dfine_engine();
  LetterboxInfo preprocess(const cv::Mat & image_bgr);
  std::vector<Detection2D> decode_yolo(
    const cv::Size & original_size,
    const LetterboxInfo & letterbox) const;
  std::vector<Detection2D> decode_dfine(const cv::Size & original_size) const;

  TensorRtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{nullptr};
  Backend backend_{Backend::kYolo};
  void * device_image_input_{nullptr};
  void * device_yolo_output_{nullptr};
  void * device_orig_target_sizes_{nullptr};
  void * device_labels_{nullptr};
  void * device_boxes_{nullptr};
  void * device_scores_{nullptr};
  std::string input_name_;
  std::string output_name_;
  int input_width_{0};
  int input_height_{0};
  int class_count_{0};
  int candidate_count_{0};
  float confidence_threshold_;
  float nms_iou_threshold_;
  std::vector<float> input_buffer_;
  std::vector<float> output_buffer_;
  std::array<std::int64_t, 2> orig_target_sizes_{};
  std::vector<std::int64_t> labels_buffer_;
  std::vector<float> boxes_buffer_;
  std::vector<float> scores_buffer_;
  cv::Ptr<cv::freetype::FreeType2> annotation_font_;
  std::string annotation_font_path_;
  int annotation_font_face_index_{-1};
};

}  // namespace yolo_det

#endif  // YOLO_DET__TENSORRT_DETECTOR_H_
