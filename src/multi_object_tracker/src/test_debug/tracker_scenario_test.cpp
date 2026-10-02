#include "tracker_scenario_test.h"

namespace multi_object_tracker::test_debug
{

TrackerDetection detection(
  double x_min,
  double y_min,
  double x_max,
  double y_max,
  std::int32_t class_id,
  double confidence)
{
  return TrackerDetection{Box2D{x_min, y_min, x_max, y_max}, class_id, confidence};
}

void require(bool condition, const std::string & message)
{
  if (!condition)
  {
    throw std::runtime_error(message);
  }
}

void test_multiple_same_class_tracks()
{
  TrackerConfig config;
  config.min_confirmed_hits = 1;
  config.confidence_score_fusion_weight = 0.0;
  ByteTracker tracker(config);

  const auto first = tracker.update(
    {
      detection(10.0, 50.0, 70.0, 170.0, 0, 0.95),
      detection(210.0, 50.0, 270.0, 170.0, 0, 0.94)
    },
    0.000);
  require(first.size() == 2, "two objects did not create two independent tracks");
  const std::uint64_t left_id = first[0].track_id;
  const std::uint64_t right_id = first[1].track_id;
  require(left_id != right_id, "two objects received the same track id");

  const auto second = tracker.update(
    {
      detection(18.0, 50.0, 78.0, 170.0, 0, 0.93),
      detection(202.0, 50.0, 262.0, 170.0, 0, 0.92)
    },
    0.033);
  require(second.size() == 2, "two-object association lost a track");
  require(second[0].track_id == left_id, "left object id changed during association");
  require(second[1].track_id == right_id, "right object id changed during association");
}

void test_two_stage_association_and_occlusion()
{
  TrackerConfig config;
  config.min_confirmed_hits = 2;
  config.max_lost_time_sec = 0.5;
  ByteTracker tracker(config);

  const auto first = tracker.update(
    {detection(10.0, 20.0, 110.0, 220.0, 0, 0.90)}, 1.000);
  require(first.empty(), "a one-frame track must remain tentative");

  const auto second = tracker.update(
    {detection(14.0, 20.0, 114.0, 220.0, 0, 0.88)}, 1.033);
  require(second.size() == 1, "the second consistent detection must confirm a track");
  const std::uint64_t stable_id = second.front().track_id;

  const auto low_confidence = tracker.update(
    {detection(18.0, 20.0, 118.0, 220.0, 0, 0.25)}, 1.066);
  require(low_confidence.size() == 1, "the low-confidence ByteTrack pass must recover the track");
  require(
    low_confidence.front().track_id == stable_id,
    "low-confidence association changed the track id");

  const auto occluded = tracker.update({}, 1.099);
  require(occluded.empty(), "an unobserved prediction must not be published");

  const auto recovered = tracker.update(
    {detection(26.0, 20.0, 126.0, 220.0, 0, 0.92)}, 1.132);
  require(recovered.size() == 1, "the track must recover after a short occlusion");
  require(recovered.front().track_id == stable_id, "short occlusion changed the track id");
}

void test_class_aware_association()
{
  TrackerConfig config;
  config.min_confirmed_hits = 1;
  ByteTracker tracker(config);

  const auto person = tracker.update(
    {detection(100.0, 100.0, 180.0, 260.0, 0, 0.95)}, 2.000);
  require(person.size() == 1, "the initial person track was not created");

  const auto bicycle = tracker.update(
    {detection(102.0, 100.0, 182.0, 260.0, 1, 0.95)}, 2.033);
  require(bicycle.size() == 1, "the overlapping second class was not created");
  require(
    bicycle.front().track_id != person.front().track_id,
    "class-aware matching reused an id across different classes");
}

void test_timestamp_gap_reset()
{
  TrackerConfig config;
  config.min_confirmed_hits = 1;
  config.reset_after_gap_sec = 1.0;
  ByteTracker tracker(config);

  const auto before_gap = tracker.update(
    {detection(40.0, 40.0, 90.0, 140.0, 2, 0.90)}, 5.000);
  require(before_gap.size() == 1, "the pre-gap track was not created");

  const auto after_gap = tracker.update(
    {detection(41.0, 40.0, 91.0, 140.0, 2, 0.90)}, 7.000);
  require(after_gap.size() == 1, "the post-gap track was not created");
  require(
    after_gap.front().track_id != before_gap.front().track_id,
    "a large timestamp gap did not reset track identity");
}

}  // namespace multi_object_tracker::test_debug

int main()
{
  try
  {
    multi_object_tracker::test_debug::test_multiple_same_class_tracks();
    multi_object_tracker::test_debug::test_two_stage_association_and_occlusion();
    multi_object_tracker::test_debug::test_class_aware_association();
    multi_object_tracker::test_debug::test_timestamp_gap_reset();
  }
  catch (const std::exception & error)
  {
    std::cerr << "tracker_scenario_test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "tracker_scenario_test passed\n";
  return 0;
}
