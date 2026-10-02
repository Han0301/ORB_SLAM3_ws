#ifndef MULTI_OBJECT_TRACKER__TRACKER_SCENARIO_TEST_H_
#define MULTI_OBJECT_TRACKER__TRACKER_SCENARIO_TEST_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "byte_tracker.h"

namespace multi_object_tracker::test_debug
{

TrackerDetection detection(
  double x_min,
  double y_min,
  double x_max,
  double y_max,
  std::int32_t class_id,
  double confidence);
void require(bool condition, const std::string & message);
void test_multiple_same_class_tracks();
void test_two_stage_association_and_occlusion();
void test_class_aware_association();
void test_timestamp_gap_reset();

}  // namespace multi_object_tracker::test_debug

#endif  // MULTI_OBJECT_TRACKER__TRACKER_SCENARIO_TEST_H_
