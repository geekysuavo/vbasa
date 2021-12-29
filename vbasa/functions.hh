
#pragma once

namespace vbasa {

/* create_model(): create a new model from point cloud data.
 *
 * arguments:
 *  @target_alpha, @target_beta: target point clouds.
 *  @source_alpha, @source_beta: source point clouds.
 *  @prior_noise_shape: prior shape parameter of the noise precision.
 *  @prior_noise_rate: prior rate parameter of the noise precision.
 *
 * returns:
 *  newly created and initialized model.
 */
vbasa::Model create_model(
  Eigen::MatrixXd target_alpha,
  Eigen::MatrixXd target_beta,
  Eigen::MatrixXd source_alpha,
  Eigen::MatrixXd source_beta,
  double prior_noise_shape = 0.001,
  double prior_noise_rate = 0.001,
  double prior_precision = 0.001
) {
  vbasa::Model mdl(prior_noise_shape, prior_noise_rate, prior_precision);
  mdl.observe(target_alpha, target_beta, source_alpha, source_beta);
  return mdl;
}

} /* namespace vbasa */

