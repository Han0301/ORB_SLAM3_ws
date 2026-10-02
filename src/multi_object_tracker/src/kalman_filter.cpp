#include "kalman_filter.h"

namespace multi_object_tracker
{

KalmanBoxFilter::KalmanBoxFilter(
  const Box2D & initial_box,
  double position_noise_weight,
  double velocity_noise_weight)
: position_noise_weight_(position_noise_weight),
  velocity_noise_weight_(velocity_noise_weight)
{
  const MeasurementVector measurement = measurement_from_box(initial_box);
  state_.head<4>() = measurement;

  const double scale = std::max(1.0, measurement(3));
  const double position_std = 2.0 * position_noise_weight_ * scale;
  const double velocity_std = 10.0 * velocity_noise_weight_ * scale;
  covariance_.diagonal() <<
    position_std * position_std,
    position_std * position_std,
    position_std * position_std,
    position_std * position_std,
    velocity_std * velocity_std,
    velocity_std * velocity_std,
    velocity_std * velocity_std,
    velocity_std * velocity_std;
}

void KalmanBoxFilter::predict(double delta_time_sec)
{
  const double delta_time = std::max(1.0e-3, delta_time_sec);
  StateMatrix transition = StateMatrix::Identity();
  for (int index = 0; index < 4; ++index)
  {
    transition(index, index + 4) = delta_time;
  }

  const double scale = std::max(1.0, state_(3));
  const double position_std = position_noise_weight_ * scale * delta_time;
  const double velocity_std = velocity_noise_weight_ * scale * std::sqrt(delta_time);
  StateMatrix process_noise = StateMatrix::Zero();
  for (int index = 0; index < 4; ++index)
  {
    process_noise(index, index) = position_std * position_std;
    process_noise(index + 4, index + 4) = velocity_std * velocity_std;
  }

  state_ = transition * state_;
  covariance_ = transition * covariance_ * transition.transpose() + process_noise;
  state_(2) = std::max(1.0, state_(2));
  state_(3) = std::max(1.0, state_(3));
}

void KalmanBoxFilter::update(const Box2D & measured_box)
{
  MeasurementMatrix observation = MeasurementMatrix::Zero();
  observation.block<4, 4>(0, 0) = Eigen::Matrix4d::Identity();

  const MeasurementVector measurement = measurement_from_box(measured_box);
  const MeasurementVector innovation = measurement - observation * state_;
  const MeasurementCovariance projected_covariance =
    observation * covariance_ * observation.transpose() + measurement_covariance();
  const Eigen::LDLT<MeasurementCovariance> decomposition(projected_covariance);
  const Eigen::Matrix<double, 8, 4> kalman_gain =
    decomposition.solve(observation * covariance_).transpose();

  state_ += kalman_gain * innovation;
  const StateMatrix identity = StateMatrix::Identity();
  const StateMatrix residual = identity - kalman_gain * observation;
  covariance_ = residual * covariance_ * residual.transpose() +
    kalman_gain * measurement_covariance() * kalman_gain.transpose();
  state_(2) = std::max(1.0, state_(2));
  state_(3) = std::max(1.0, state_(3));
}

double KalmanBoxFilter::gating_distance(const Box2D & measured_box) const
{
  MeasurementMatrix observation = MeasurementMatrix::Zero();
  observation.block<4, 4>(0, 0) = Eigen::Matrix4d::Identity();

  const MeasurementVector innovation = measurement_from_box(measured_box) -
    observation * state_;
  const MeasurementCovariance projected_covariance =
    observation * covariance_ * observation.transpose() + measurement_covariance();
  const Eigen::LDLT<MeasurementCovariance> decomposition(projected_covariance);
  return innovation.dot(decomposition.solve(innovation));
}

Box2D KalmanBoxFilter::box() const
{
  const double half_width = std::max(1.0, state_(2)) * 0.5;
  const double half_height = std::max(1.0, state_(3)) * 0.5;
  return Box2D{
    state_(0) - half_width,
    state_(1) - half_height,
    state_(0) + half_width,
    state_(1) + half_height};
}

KalmanBoxFilter::MeasurementVector KalmanBoxFilter::measurement_from_box(
  const Box2D & box)
{
  const double width = std::max(1.0, box.x_max - box.x_min);
  const double height = std::max(1.0, box.y_max - box.y_min);
  MeasurementVector measurement;
  measurement <<
    box.x_min + width * 0.5,
    box.y_min + height * 0.5,
    width,
    height;
  return measurement;
}

KalmanBoxFilter::MeasurementCovariance KalmanBoxFilter::measurement_covariance() const
{
  const double scale = std::max(1.0, state_(3));
  const double measurement_std = position_noise_weight_ * scale;
  return MeasurementCovariance::Identity() * measurement_std * measurement_std;
}

}  // namespace multi_object_tracker
