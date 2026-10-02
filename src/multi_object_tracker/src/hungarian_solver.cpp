#include "hungarian_solver.h"

namespace multi_object_tracker
{

AssignmentResult HungarianSolver::solve(
  const std::vector<std::vector<double>> & cost_matrix,
  double maximum_cost)
{
  AssignmentResult result;
  const std::size_t row_count = cost_matrix.size();
  const std::size_t column_count = row_count == 0 ? 0 : cost_matrix.front().size();

  if (row_count == 0)
  {
    for (std::size_t column = 0; column < column_count; ++column)
    {
      result.unmatched_columns.push_back(column);
    }
    return result;
  }
  if (column_count == 0)
  {
    for (std::size_t row = 0; row < row_count; ++row)
    {
      result.unmatched_rows.push_back(row);
    }
    return result;
  }

  const std::size_t dimension = std::max(row_count, column_count);
  const double dummy_cost = maximum_cost + 1.0e-6;
  const double invalid_cost = maximum_cost + 1.0e6;
  std::vector<std::vector<double>> square_cost(
    dimension, std::vector<double>(dimension, 0.0));

  for (std::size_t row = 0; row < dimension; ++row)
  {
    for (std::size_t column = 0; column < dimension; ++column)
    {
      if (row < row_count && column < column_count)
      {
        const double input_cost = cost_matrix[row][column];
        square_cost[row][column] = input_cost <= maximum_cost ? input_cost : invalid_cost;
      }
      else if (row < row_count || column < column_count)
      {
        square_cost[row][column] = dummy_cost;
      }
    }
  }

  std::vector<double> row_potential(dimension + 1, 0.0);
  std::vector<double> column_potential(dimension + 1, 0.0);
  std::vector<std::size_t> column_match(dimension + 1, 0);
  std::vector<std::size_t> previous_column(dimension + 1, 0);

  for (std::size_t row = 1; row <= dimension; ++row)
  {
    column_match[0] = row;
    std::size_t current_column = 0;
    std::vector<double> minimum_value(
      dimension + 1, std::numeric_limits<double>::infinity());
    std::vector<bool> used(dimension + 1, false);

    do
    {
      used[current_column] = true;
      const std::size_t current_row = column_match[current_column];
      double delta = std::numeric_limits<double>::infinity();
      std::size_t next_column = 0;

      for (std::size_t column = 1; column <= dimension; ++column)
      {
        if (used[column])
        {
          continue;
        }
        const double reduced_cost = square_cost[current_row - 1][column - 1] -
          row_potential[current_row] - column_potential[column];
        if (reduced_cost < minimum_value[column])
        {
          minimum_value[column] = reduced_cost;
          previous_column[column] = current_column;
        }
        if (minimum_value[column] < delta)
        {
          delta = minimum_value[column];
          next_column = column;
        }
      }

      for (std::size_t column = 0; column <= dimension; ++column)
      {
        if (used[column])
        {
          row_potential[column_match[column]] += delta;
          column_potential[column] -= delta;
        }
        else
        {
          minimum_value[column] -= delta;
        }
      }
      current_column = next_column;
    }
    while (column_match[current_column] != 0);

    do
    {
      const std::size_t next_column = previous_column[current_column];
      column_match[current_column] = column_match[next_column];
      current_column = next_column;
    }
    while (current_column != 0);
  }

  std::vector<bool> row_matched(row_count, false);
  std::vector<bool> column_matched(column_count, false);
  for (std::size_t column = 1; column <= dimension; ++column)
  {
    const std::size_t row = column_match[column];
    if (row == 0 || row > row_count || column > column_count)
    {
      continue;
    }
    const double cost = cost_matrix[row - 1][column - 1];
    if (cost <= maximum_cost)
    {
      result.matches.emplace_back(row - 1, column - 1);
      row_matched[row - 1] = true;
      column_matched[column - 1] = true;
    }
  }

  for (std::size_t row = 0; row < row_count; ++row)
  {
    if (!row_matched[row])
    {
      result.unmatched_rows.push_back(row);
    }
  }
  for (std::size_t column = 0; column < column_count; ++column)
  {
    if (!column_matched[column])
    {
      result.unmatched_columns.push_back(column);
    }
  }
  return result;
}

}  // namespace multi_object_tracker
