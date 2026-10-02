#ifndef MULTI_OBJECT_TRACKER__BYTE_TRACKER_H_
#define MULTI_OBJECT_TRACKER__BYTE_TRACKER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hungarian_solver.h"
#include "kalman_filter.h"
#include "tracker_types.h"

namespace multi_object_tracker
{

class ByteTracker
{
public:
  explicit ByteTracker(const TrackerConfig & config);

  std::vector<TrackedObject> update(
    const std::vector<TrackerDetection> & detections,
    double timestamp_sec);
  void reset();
  std::size_t active_track_count() const;

private:
  enum class TrackState
  {
    Tentative,
    Confirmed,
    Lost,
    Removed
  };

  struct Track
  {
    Track(
      const TrackerDetection & detection,
      std::uint64_t identifier,
      double timestamp_sec,
      const TrackerConfig & config);

    std::uint64_t id{0};
    std::int32_t class_id{0};
    double confidence{0.0};
    KalmanBoxFilter filter;
    TrackState state{TrackState::Tentative};
    std::uint32_t hits{1};
    std::uint32_t age{1};
    double last_detection_timestamp_sec{0.0};
    bool updated_this_frame{true};
  };

  struct Association
  {
    std::vector<std::pair<std::size_t, std::size_t>> matches;
    std::vector<std::size_t> unmatched_tracks;
    std::vector<std::size_t> unmatched_detections;
  };

  Association associate(
    const std::vector<std::size_t> & track_indices,
    const std::vector<TrackerDetection> & detections,
    const std::vector<std::size_t> & detection_indices,
    double minimum_iou,
    bool fuse_detection_score) const;
  void update_track(
    Track & track,
    const TrackerDetection & detection,
    double timestamp_sec);
  static double intersection_over_union(const Box2D & first, const Box2D & second);
  void validate_config() const;

  TrackerConfig config_;
  std::vector<Track> tracks_;
  std::uint64_t next_track_id_{1};
  double last_timestamp_sec_{std::numeric_limits<double>::quiet_NaN()};
};

}  // namespace multi_object_tracker

#endif  // MULTI_OBJECT_TRACKER__BYTE_TRACKER_H_
