#ifndef MULTI_OBJECT_TRACKER__KALMAN_FILTER_H_
#define MULTI_OBJECT_TRACKER__KALMAN_FILTER_H_

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "tracker_types.h"

namespace multi_object_tracker
{

class KalmanBoxFilter
{
public:
  KalmanBoxFilter(
    const Box2D & initial_box,
    double position_noise_weight,
    double velocity_noise_weight);

  void predict(double delta_time_sec);
  void update(const Box2D & measured_box);
  double gating_distance(const Box2D & measured_box) const;
  Box2D box() const;

private:
  using StateVector = Eigen::Matrix<double, 8, 1>;
  using StateMatrix = Eigen::Matrix<double, 8, 8>;
  using MeasurementVector = Eigen::Matrix<double, 4, 1>;
  using MeasurementMatrix = Eigen::Matrix<double, 4, 8>;
  using MeasurementCovariance = Eigen::Matrix<double, 4, 4>;

  static MeasurementVector measurement_from_box(const Box2D & box);
  MeasurementCovariance measurement_covariance() const;

  StateVector state_{StateVector::Zero()};
  StateMatrix covariance_{StateMatrix::Zero()};
  double position_noise_weight_{0.05};
  double velocity_noise_weight_{0.00625};
};

}  // namespace multi_object_tracker

#endif  // MULTI_OBJECT_TRACKER__KALMAN_FILTER_H_
