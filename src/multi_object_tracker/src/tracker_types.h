#ifndef MULTI_OBJECT_TRACKER__TRACKER_TYPES_H_
#define MULTI_OBJECT_TRACKER__TRACKER_TYPES_H_

#include <cstdint>

namespace multi_object_tracker
{

struct Box2D
{
  double x_min{0.0};
  double y_min{0.0};
  double x_max{0.0};
  double y_max{0.0};
};

struct TrackerDetection
{
  Box2D box;
  std::int32_t class_id{0};
  double confidence{0.0};
};

struct TrackedObject
{
  std::uint64_t track_id{0};
  Box2D box;
  std::int32_t class_id{0};
  double confidence{0.0};
};

struct TrackerConfig
{
  double high_confidence_threshold{0.60};
  double low_confidence_threshold{0.10};
  double new_track_threshold{0.70};
  double high_match_iou_threshold{0.30};
  double low_match_iou_threshold{0.20};
  double tentative_match_iou_threshold{0.30};
  double confidence_score_fusion_weight{0.10};
  double mahalanobis_threshold{13.2767};
  std::uint32_t min_confirmed_hits{2};
  double max_lost_time_sec{0.50};
  double expected_frame_rate{30.0};
  double max_dt_sec{0.20};
  double reset_after_gap_sec{1.00};
  bool class_aware_association{true};
  double kalman_position_noise_weight{0.05};
  double kalman_velocity_noise_weight{0.00625};
};

}  // namespace multi_object_tracker

#endif  // MULTI_OBJECT_TRACKER__TRACKER_TYPES_H_
