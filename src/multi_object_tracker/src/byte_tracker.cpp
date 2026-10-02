#include "byte_tracker.h"

namespace multi_object_tracker
{

ByteTracker::Track::Track(
  const TrackerDetection & detection,
  std::uint64_t identifier,
  double timestamp_sec,
  const TrackerConfig & config)
: id(identifier),
  class_id(detection.class_id),
  confidence(detection.confidence),
  filter(
    detection.box,
    config.kalman_position_noise_weight,
    config.kalman_velocity_noise_weight),
  state(config.min_confirmed_hits <= 1 ? TrackState::Confirmed : TrackState::Tentative),
  last_detection_timestamp_sec(timestamp_sec)
{
}

ByteTracker::ByteTracker(const TrackerConfig & config)
: config_(config)
{
  validate_config();
}

std::vector<TrackedObject> ByteTracker::update(
  const std::vector<TrackerDetection> & detections,
  double timestamp_sec)
{
  if (!std::isfinite(timestamp_sec))
  {
    timestamp_sec = std::isfinite(last_timestamp_sec_) ?
      last_timestamp_sec_ + 1.0 / config_.expected_frame_rate : 0.0;
  }

  double delta_time_sec = 1.0 / config_.expected_frame_rate;
  if (std::isfinite(last_timestamp_sec_))
  {
    const double raw_delta_time = timestamp_sec - last_timestamp_sec_;
    if (raw_delta_time <= 0.0 || raw_delta_time > config_.reset_after_gap_sec)
    {
      reset();
    }
    else
    {
      delta_time_sec = std::min(raw_delta_time, config_.max_dt_sec);
    }
  }
  last_timestamp_sec_ = timestamp_sec;

  for (Track & track : tracks_)
  {
    track.filter.predict(delta_time_sec);
    track.updated_this_frame = false;
    ++track.age;
  }

  std::vector<std::size_t> high_detections;
  std::vector<std::size_t> low_detections;
  for (std::size_t index = 0; index < detections.size(); ++index)
  {
    if (detections[index].confidence >= config_.high_confidence_threshold)
    {
      high_detections.push_back(index);
    }
    else if (detections[index].confidence >= config_.low_confidence_threshold)
    {
      low_detections.push_back(index);
    }
  }

  std::vector<std::size_t> established_tracks;
  std::vector<std::size_t> tentative_tracks;
  for (std::size_t index = 0; index < tracks_.size(); ++index)
  {
    if (tracks_[index].state == TrackState::Tentative)
    {
      tentative_tracks.push_back(index);
    }
    else if (tracks_[index].state == TrackState::Confirmed ||
      tracks_[index].state == TrackState::Lost)
    {
      established_tracks.push_back(index);
    }
  }

  const Association high_association = associate(
    established_tracks,
    detections,
    high_detections,
    config_.high_match_iou_threshold,
    true);
  for (const auto & match : high_association.matches)
  {
    update_track(tracks_[match.first], detections[match.second], timestamp_sec);
  }

  std::vector<std::size_t> unmatched_confirmed_tracks;
  for (const std::size_t track_index : high_association.unmatched_tracks)
  {
    if (tracks_[track_index].state == TrackState::Confirmed)
    {
      unmatched_confirmed_tracks.push_back(track_index);
    }
  }

  const Association low_association = associate(
    unmatched_confirmed_tracks,
    detections,
    low_detections,
    config_.low_match_iou_threshold,
    false);
  for (const auto & match : low_association.matches)
  {
    update_track(tracks_[match.first], detections[match.second], timestamp_sec);
  }
  for (const std::size_t track_index : low_association.unmatched_tracks)
  {
    tracks_[track_index].state = TrackState::Lost;
  }

  const Association tentative_association = associate(
    tentative_tracks,
    detections,
    high_association.unmatched_detections,
    config_.tentative_match_iou_threshold,
    true);
  for (const auto & match : tentative_association.matches)
  {
    update_track(tracks_[match.first], detections[match.second], timestamp_sec);
  }
  for (const std::size_t track_index : tentative_association.unmatched_tracks)
  {
    tracks_[track_index].state = TrackState::Removed;
  }

  for (const std::size_t detection_index : tentative_association.unmatched_detections)
  {
    const TrackerDetection & detection = detections[detection_index];
    if (detection.confidence >= config_.new_track_threshold)
    {
      tracks_.emplace_back(
        detection,
        next_track_id_++,
        timestamp_sec,
        config_);
    }
  }

  for (Track & track : tracks_)
  {
    if (track.state == TrackState::Lost &&
      timestamp_sec - track.last_detection_timestamp_sec > config_.max_lost_time_sec)
    {
      track.state = TrackState::Removed;
    }
  }

  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [](const Track & track)
      {
        return track.state == TrackState::Removed;
      }),
    tracks_.end());

  std::vector<TrackedObject> output;
  for (const Track & track : tracks_)
  {
    if (track.state == TrackState::Confirmed && track.updated_this_frame)
    {
      output.push_back(TrackedObject{
        track.id,
        track.filter.box(),
        track.class_id,
        track.confidence});
    }
  }
  return output;
}

void ByteTracker::reset()
{
  tracks_.clear();
  last_timestamp_sec_ = std::numeric_limits<double>::quiet_NaN();
}

std::size_t ByteTracker::active_track_count() const
{
  return tracks_.size();
}

ByteTracker::Association ByteTracker::associate(
  const std::vector<std::size_t> & track_indices,
  const std::vector<TrackerDetection> & detections,
  const std::vector<std::size_t> & detection_indices,
  double minimum_iou,
  bool fuse_detection_score) const
{
  Association association;
  if (track_indices.empty())
  {
    association.unmatched_detections = detection_indices;
    return association;
  }
  if (detection_indices.empty())
  {
    association.unmatched_tracks = track_indices;
    return association;
  }

  const double maximum_cost = 1.0 - minimum_iou;
  std::vector<std::vector<double>> costs(
    track_indices.size(),
    std::vector<double>(detection_indices.size(), maximum_cost + 1.0));

  for (std::size_t row = 0; row < track_indices.size(); ++row)
  {
    const Track & track = tracks_[track_indices[row]];
    for (std::size_t column = 0; column < detection_indices.size(); ++column)
    {
      const TrackerDetection & detection = detections[detection_indices[column]];
      if (config_.class_aware_association && track.class_id != detection.class_id)
      {
        continue;
      }
      if (track.filter.gating_distance(detection.box) > config_.mahalanobis_threshold)
      {
        continue;
      }

      const double iou = intersection_over_union(track.filter.box(), detection.box);
      double similarity = iou;
      if (fuse_detection_score)
      {
        const double score_factor = 1.0 - config_.confidence_score_fusion_weight +
          config_.confidence_score_fusion_weight * detection.confidence;
        similarity *= score_factor;
      }
      if (similarity >= minimum_iou)
      {
        costs[row][column] = 1.0 - similarity;
      }
    }
  }

  const AssignmentResult result = HungarianSolver::solve(costs, maximum_cost);
  for (const auto & match : result.matches)
  {
    association.matches.emplace_back(
      track_indices[match.first], detection_indices[match.second]);
  }
  for (const std::size_t row : result.unmatched_rows)
  {
    association.unmatched_tracks.push_back(track_indices[row]);
  }
  for (const std::size_t column : result.unmatched_columns)
  {
    association.unmatched_detections.push_back(detection_indices[column]);
  }
  return association;
}

void ByteTracker::update_track(
  Track & track,
  const TrackerDetection & detection,
  double timestamp_sec)
{
  track.filter.update(detection.box);
  track.class_id = detection.class_id;
  track.confidence = detection.confidence;
  track.last_detection_timestamp_sec = timestamp_sec;
  track.updated_this_frame = true;
  ++track.hits;
  if (track.hits >= config_.min_confirmed_hits)
  {
    track.state = TrackState::Confirmed;
  }
}

double ByteTracker::intersection_over_union(const Box2D & first, const Box2D & second)
{
  const double intersection_width = std::max(
    0.0, std::min(first.x_max, second.x_max) - std::max(first.x_min, second.x_min));
  const double intersection_height = std::max(
    0.0, std::min(first.y_max, second.y_max) - std::max(first.y_min, second.y_min));
  const double intersection = intersection_width * intersection_height;
  const double first_area = std::max(0.0, first.x_max - first.x_min) *
    std::max(0.0, first.y_max - first.y_min);
  const double second_area = std::max(0.0, second.x_max - second.x_min) *
    std::max(0.0, second.y_max - second.y_min);
  const double union_area = first_area + second_area - intersection;
  return union_area > 0.0 ? intersection / union_area : 0.0;
}

void ByteTracker::validate_config() const
{
  const auto valid_probability = [](double value)
    {
      return value >= 0.0 && value <= 1.0;
    };
  if (!valid_probability(config_.high_confidence_threshold) ||
    !valid_probability(config_.low_confidence_threshold) ||
    !valid_probability(config_.new_track_threshold) ||
    config_.low_confidence_threshold > config_.high_confidence_threshold ||
    config_.new_track_threshold < config_.high_confidence_threshold)
  {
    throw std::invalid_argument("invalid ByteTrack confidence thresholds");
  }
  if (!valid_probability(config_.high_match_iou_threshold) ||
    !valid_probability(config_.low_match_iou_threshold) ||
    !valid_probability(config_.tentative_match_iou_threshold) ||
    !valid_probability(config_.confidence_score_fusion_weight))
  {
    throw std::invalid_argument("invalid association thresholds");
  }
  if (config_.min_confirmed_hits == 0 || config_.max_lost_time_sec <= 0.0 ||
    config_.expected_frame_rate <= 0.0 || config_.max_dt_sec <= 0.0 ||
    config_.reset_after_gap_sec <= config_.max_dt_sec ||
    config_.mahalanobis_threshold <= 0.0 ||
    config_.kalman_position_noise_weight <= 0.0 ||
    config_.kalman_velocity_noise_weight <= 0.0)
  {
    throw std::invalid_argument("invalid tracker timing or Kalman parameters");
  }
}

}  // namespace multi_object_tracker
