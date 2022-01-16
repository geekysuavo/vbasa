
#pragma once

namespace vbasa {

/* create_model(): create a new model from point cloud data.
 *
 * arguments:
 *  @target_alpha, @target_beta: target point clouds.
 *  @source_alpha, @source_beta: source point clouds.
 *  @target_residues: target amino acid residues.
 *  @source_residues: source amino acid resodues.
 *  @prior_noise_shape: prior shape parameter of the noise precision.
 *  @prior_noise_rate: prior rate parameter of the noise precision.
 *  @prior_diag: prior diagonal value of the rotation matrix parameter.
 *  @sub_matrix: optional substitution matrix for computing matching priors.
 *
 * returns:
 *  newly created and initialized model.
 */
vbasa::Model create_model(
  Eigen::MatrixXd target_alpha,
  Eigen::MatrixXd target_beta,
  Eigen::MatrixXd source_alpha,
  Eigen::MatrixXd source_beta,
  const vbasa::SubstitutionMatrix::sequence_t& target_sequence,
  const vbasa::SubstitutionMatrix::sequence_t& source_sequence,
  double prior_noise_shape = 1e-3,
  double prior_noise_rate = 1e-3,
  double prior_precision = 1e-4,
  double prior_diag = 1e-6,
  std::optional<vbasa::SubstitutionMatrix> sub_matrix = {}
) {
  vbasa::Model mdl(
    prior_noise_shape,
    prior_noise_rate,
    prior_precision,
    prior_diag
  );

  std::optional<Eigen::MatrixXd> omega0;
  if (sub_matrix) {
    auto sub_matrix_obj = sub_matrix.value();
    omega0 = sub_matrix_obj.eval(target_sequence, source_sequence);
  }

  mdl.observe(target_alpha, target_beta, source_alpha, source_beta, omega0);
  return mdl;
}

} /* namespace vbasa */

