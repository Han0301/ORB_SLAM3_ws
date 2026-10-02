#ifndef MULTI_OBJECT_TRACKER__HUNGARIAN_SOLVER_H_
#define MULTI_OBJECT_TRACKER__HUNGARIAN_SOLVER_H_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace multi_object_tracker
{

struct AssignmentResult
{
  std::vector<std::pair<std::size_t, std::size_t>> matches;
  std::vector<std::size_t> unmatched_rows;
  std::vector<std::size_t> unmatched_columns;
};

class HungarianSolver
{
public:
  static AssignmentResult solve(
    const std::vector<std::vector<double>> & cost_matrix,
    double maximum_cost);
};

}  // namespace multi_object_tracker

#endif  // MULTI_OBJECT_TRACKER__HUNGARIAN_SOLVER_H_
