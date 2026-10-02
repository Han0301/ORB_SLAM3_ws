#ifndef YOLO_DET__DETECTION_TYPES_H_
#define YOLO_DET__DETECTION_TYPES_H_

#include <cstdint>

namespace yolo_det
{

struct BoundingBox2D
{
  float x_min;
  float y_min;
  float x_max;
  float y_max;
};

struct Detection2D
{
  BoundingBox2D box;
  int32_t class_id;
  float confidence;
};

}  // namespace yolo_det

#endif  // YOLO_DET__DETECTION_TYPES_H_
