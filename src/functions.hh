
#pragma once

namespace vbasa {

/* initial_rotations(): return all rotations from the octahedral group,
 * which are distributed evenly enough for multistart model fitting.
 */
std::vector<Eigen::MatrixXd> initial_rotations(unsigned int k = 3) {
  std::vector<Eigen::MatrixXd> rotations;
  std::vector<unsigned int> perm(k);

  for (unsigned int i = 0; i < k; ++i)
    perm[i] = i;

  do {
    for (unsigned int signs = 0; signs < 8; ++signs) {
      Eigen::MatrixXd R = Eigen::MatrixXd::Zero(k, k);
      for (unsigned int i = 0; i < k; ++i)
        R(i, perm[i]) = (signs >> i) & 1 ? -1.0 : 1.0;

      if (R.determinant() > 0)
        rotations.push_back(R);
    }
  } while (std::next_permutation(perm.begin(), perm.end()));

  return rotations;
}

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
 *  @prior_unmatched: prior probability that a target point pair is unmatched.
 *  @argmax_rotation: whether to approximate q(R) as a delta function at its mode.
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
  double prior_noise_shape = 1.0,
  double prior_noise_rate = 1.0,
  double prior_precision = 1e-4,
  double prior_diag = 1e-6,
  double prior_unmatched = 0.1,
  bool argmax_rotation = false,
  std::optional<vbasa::SubstitutionMatrix> sub_matrix = {}
) {
  vbasa::Model mdl(
    prior_noise_shape,
    prior_noise_rate,
    prior_precision,
    prior_diag,
    prior_unmatched,
    argmax_rotation
  );

  std::optional<Eigen::MatrixXd> omega0;
  if (sub_matrix)
    omega0 = sub_matrix->eval(target_sequence, source_sequence);

  mdl.observe(target_alpha, target_beta, source_alpha, source_beta, omega0);
  return mdl;
}

/* multistart_model(): fit multiple models from different initial rotations,
 * and return the model with the largest variational lower bound.
 *
 * arguments:
 *  same as create_model().
 *
 * returns:
 *  newly created, initialized, and optimized model.
 */
vbasa::Model multistart_model(
  Eigen::MatrixXd target_alpha,
  Eigen::MatrixXd target_beta,
  Eigen::MatrixXd source_alpha,
  Eigen::MatrixXd source_beta,
  const vbasa::SubstitutionMatrix::sequence_t& target_sequence,
  const vbasa::SubstitutionMatrix::sequence_t& source_sequence,
  double prior_noise_shape = 1.0,
  double prior_noise_rate = 1.0,
  double prior_precision = 1e-4,
  double prior_diag = 1e-6,
  double prior_unmatched = 0.1,
  bool argmax_rotation = false,
  std::optional<vbasa::SubstitutionMatrix> sub_matrix = {}
) {
  std::optional<Eigen::MatrixXd> omega0;
  if (sub_matrix)
    omega0 = sub_matrix->eval(target_sequence, source_sequence);

  std::optional<vbasa::Model> best_model;
  double best_elbo = -std::numeric_limits<double>::infinity();

  const unsigned int k = target_alpha.cols();
  for (const auto& R0 : initial_rotations(k)) {
    vbasa::Model mdl(
      prior_noise_shape,
      prior_noise_rate,
      prior_precision,
      prior_diag,
      prior_unmatched,
      argmax_rotation
    );

    mdl.observe(target_alpha, target_beta, source_alpha, source_beta, omega0, R0);

    const double elbo = mdl.optimize();
    if (std::isfinite(elbo) && (!best_model || elbo > best_elbo)) {
      best_model = mdl;
      best_elbo = elbo;
    }
  }

  if (!best_model)
    throw std::runtime_error("failed to successfully fit any model");

  return *best_model;
}

} /* namespace vbasa */

