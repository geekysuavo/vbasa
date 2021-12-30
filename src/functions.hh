
#pragma once

namespace vbasa {

/* create_model(): create a new model from point cloud data.
 *
 * arguments:
 *  @target_alpha, @target_beta: target point clouds.
 *  @source_alpha, @source_beta: source point clouds.
 *  @prior_noise_shape: prior shape parameter of the noise precision.
 *  @prior_noise_rate: prior rate parameter of the noise precision.
 *  @prior_diag: prior diagonal value of the rotation matrix parameter.
 *
 * returns:
 *  newly created and initialized model.
 */
vbasa::Model create_model(
  Eigen::MatrixXd target_alpha,
  Eigen::MatrixXd target_beta,
  Eigen::MatrixXd source_alpha,
  Eigen::MatrixXd source_beta,
  double prior_noise_shape = 1e-3,
  double prior_noise_rate = 1e-3,
  double prior_precision = 1e-4,
  double prior_diag = 1e-6
) {
  vbasa::Model mdl(
    prior_noise_shape,
    prior_noise_rate,
    prior_precision,
    prior_diag
  );

  mdl.observe(target_alpha, target_beta, source_alpha, source_beta);
  return mdl;
}

} /* namespace vbasa */

