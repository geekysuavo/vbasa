
#pragma once

namespace vbasa {

class Model {
public:
  /* Model(): constructor taking configuration values.
   *
   * arguments:
   *  @prior_alpha: initial value of alpha0.
   *  @prior_beta: initial value of beta0.
   *  @prior_gamma: initial value of gamma0.
   *  @prior_rho: initial value of rho0.
   */
  Model(
    double prior_alpha,
    double prior_beta,
    double prior_gamma,
    double prior_rho
  ) {
    if (prior_alpha <= 0)
      throw std::out_of_range("prior noise shape must be positive");

    if (prior_beta <= 0)
      throw std::out_of_range("prior noise rate must be positive");

    if (prior_gamma <= 0)
      throw std::out_of_range("prior precision must be positive");

    if (prior_rho <= 0)
      throw std::out_of_range("prior matrix diag must be positive");

    alpha0 = prior_alpha;
    beta0 = prior_beta;
    gamma0 = prior_gamma;
    rho0 = prior_rho;
  }

  /* Model(): default constructor.
   */
  Model() : Model(1e-3, 1e-3, 1e-4, 1e-6) {}

  /* observe(): initialize member data given an input data set.
   *
   * arguments:
   *  @xa: [m, k], target alpha points.
   *  @xb: [m, k], target beta points.
   *  @ya: [n, k], source alpha points.
   *  @yb: [n, k], source beta points.
   */
  void observe(
    const Eigen::Ref<const Eigen::MatrixXd> xa,
    const Eigen::Ref<const Eigen::MatrixXd> xb,
    const Eigen::Ref<const Eigen::MatrixXd> ya,
    const Eigen::Ref<const Eigen::MatrixXd> yb
  ) {
    const auto m = xa.rows();
    const auto n = ya.rows();
    const auto k = xa.cols();

    if (xb.rows() != m)
      throw std::length_error("source row count mismatch");

    if (yb.rows() != n)
      throw std::length_error("target row count mismatch");

    if (xb.cols() != k || ya.cols() != k || yb.cols() != k)
      throw std::length_error("column count mismatch");

    /* store the input data. */
    Xa = xa;
    Xb = xb;
    Ya = ya;
    Yb = yb;

    /* initialize the prior parameters. */
    omega0.resize(m, n);
    A0.resize(k, k);

    /* FIXME: init omega0 */
    A0.setZero();
    A0.diagonal() = Eigen::ArrayXd::Constant(k, 1, rho0);

    /* initialize the fit parameters. */
    alpha.resize(n);
    beta.resize(n);
    eta.resize(k);
    A.resize(k, k);

    alpha.fill(alpha0);
    beta.fill(beta0);
    gamma = gamma0;
    eta.setZero();
    A = A0;

    /* initialize the expectations. */
    lambda.resize(n);
    ln_lambda.resize(n);
    t.resize(k);
    Qa.resize(m, n);
    Qb.resize(m, n);
    Z.resize(m, n);
    R.resize(k, k);

    step(/*modify=*/false);
  }

  /* step(): perform a single optimization step.
   *
   * arguments:
   *  @modify: if false, only expectations are updated.
   *
   * returns:
   *  value of the variational lower bound.
   */
  double step(bool modify = true) {
    update_rotation(modify);
    update_translation(modify);
    update_assignments(modify);
    update_precisions(modify);

    return bound();
  }

  /* optimize(): optimize variational parameters.
   *
   * arguments:
   *  @iters: number of optimization steps to perform.
   *  @th_abs: absolute threshold for early stopping.
   *  @th_rel: relative threshold for early stopping.
   *
   * returns:
   *  final value of the variational lower bound.
   */
  double optimize(
    unsigned int iters = 1000,
    double th_abs = 0.001,
    double th_rel = 0.01
  ) {
    double elbo = bound();

    for (unsigned int it = 0; it < iters; it++) {
      const double elbo_prev = elbo;
      elbo = step();

      const double change = std::abs(elbo_prev - elbo);
      const double rel_change = change / elbo_prev;

      if (change < th_abs || rel_change < th_rel)
        break;
    }

    return elbo;
  }

  /* bound(): compute the variational lower bound.
   */
  double bound() const {
    return 0; /* FIXME: write bound() */
  }

  /* num_target(): get the number of target point pairs (m).
   */
  unsigned int num_target() const {
    return Xa.rows();
  }

  /* num_source(): get the number of source point pairs (n).
   */
  unsigned int num_source() const {
    return Ya.rows();
  }

  /* dims(): get the point cloud dimensionality (k).
   */
  unsigned int dims() const {
    return Xa.cols();
  }

  /* sizes(): return a tuple holding (m, n, k).
   */
  std::tuple<unsigned int, unsigned int, unsigned int> sizes() const {
    return {num_target(), num_source(), dims()};
  }

  /* input point clouds:
   *  @Xa, @Xb: [m, k], target points.
   *  @Ya, @Yb: [n, k], source points.
   */
  Eigen::MatrixXd Xa, Xb, Ya, Yb;

  /* prior parameters:
   *  @alpha0: alignment noise shape.
   *  @beta0: alignment noise rate.
   *  @gamma0: translation precision.
   *  @rho0: rotation parameter diagonal.
   *  @omega0: [m, n], assignment weights.
   *  @A0: [k, k], rotation parameter.
   */
  double alpha0, beta0, gamma0, rho0;
  Eigen::MatrixXd omega0;
  Eigen::MatrixXd A0;

  /* parameters:
   *  @alpha: [n], alignment noise shape.
   *  @beta: [n], alignment noise rate.
   *  @eta: [k], translation shift parameter.
   *  @A: [k, k], rotation parameter.
   *  @gamma: translation precision.
   */
  Eigen::VectorXd alpha, beta, eta;
  Eigen::MatrixXd A;
  double gamma;

  /* vector-valued expectations:
   *  @lambda: [n], alignment noise precision.
   *  @ln_lambda: [n], alignment noise log-precision.
   *  @t: [k], translation first non-central moment.
   *
   * matrix-valued expectations:
   *  @Qa, @Qb: [m, n], squared alignment errors of alpha and beta.
   *  @Z: [m, n], assignment matrix.
   *  @R: [k, k], rotation matrix.
   *
   * scalar-valued expectations:
   *  @tt: translation inner product.
   */
  Eigen::VectorXd lambda, ln_lambda, t;
  Eigen::MatrixXd Qa, Qb, Z, R;
  double tt;

private:
  /* update_rotation(): update q(R), the rotation factor.
   */
  void update_rotation(bool modify) {
    if (modify) {
      /* FIXME: update A */
    }

    /* FIXME: update R */

    recompute_alignment_errors();
  }

  /* update_translation(): update q(t), the translation factor.
   */
  void update_translation(bool modify) {
    if (modify) {
      /* FIXME: update gamma, eta */
    }

    /* FIXME: update t, tt */

    recompute_alignment_errors();
  }

  /* update_assignments(): update q(Z), the assignments factor.
   */
  void update_assignments(bool modify) {
    if (modify) {
      /* FIXME: update Z */
    }
  }

  /* update_precisions(): update q(lambda), the noise precision factors.
   */
  void update_precisions(bool modify) {
    if (modify) {
      const double k = dims();
      Eigen::VectorXd z = Z.colwise().sum();
      Eigen::VectorXd zqq = (Z.array() * (Qa + Qb).array()).colwise().sum();

      alpha = alpha0 + k * z.array();
      beta = beta0 + 0.5 * zqq.array();
    }

    lambda = alpha.array() / beta.array();
    ln_lambda = alpha.array().digamma() - beta.array().log();
  }

  /* recompute_alignment_errors(): recompute Qa and Qb.
   */
  void recompute_alignment_errors() {
    recompute_alignment_errors_impl(Xa, Ya, Qa);
    recompute_alignment_errors_impl(Xb, Yb, Qb);
  }

  /* recompute_alignment_errors_impl(): recompute a Q-matrix expectation.
   */
  void recompute_alignment_errors_impl(
    const Eigen::Ref<const Eigen::MatrixXd> X,
    const Eigen::Ref<const Eigen::MatrixXd> Y,
    Eigen::Ref<Eigen::MatrixXd> Q
  ) {
    const auto [m, n, k] = sizes();
    Eigen::VectorXd x, y, Ry;

    for (unsigned int j = 0; j < n; j++) {
      y = Y.row(j);
      Ry = R * y;
      const double y2 = y.squaredNorm();

      for (unsigned int i = 0; i < m; i++) {
        x = X.row(i);
        const double x2 = x.squaredNorm();

        Q(i, j) = x2 + y2 + tt - 2 * t.dot(x - Ry) - 2 * x.dot(Ry);
      }
    }
  }
};

} /* namespace vbasa */

