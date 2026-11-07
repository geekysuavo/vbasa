
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
   *  @prior_unmatched: prior probability that a target point pair is unmatched.
   *  @argmax_rotation: whether to approximate q(R) as a delta function at its mode.
   */
  Model(
    double prior_alpha,
    double prior_beta,
    double prior_gamma,
    double prior_rho,
    double prior_unmatched = 0.1,
    bool argmax_rotation = false
  ) {
    if (prior_alpha <= 0)
      throw std::out_of_range("prior noise shape must be positive");

    if (prior_beta <= 0)
      throw std::out_of_range("prior noise rate must be positive");

    if (prior_gamma <= 0)
      throw std::out_of_range("prior precision must be positive");

    if (prior_rho <= 0)
      throw std::out_of_range("prior matrix diag must be positive");

    if (prior_unmatched < 0 || prior_unmatched >= 1)
      throw std::out_of_range("prior unmatched probability must be in [0,1)");

    alpha0 = prior_alpha;
    beta0 = prior_beta;
    gamma0 = prior_gamma;
    rho0 = prior_rho;
    w0 = prior_unmatched;
    use_argmax_rotation = argmax_rotation;
  }

  /* Model(): default constructor.
   */
  Model() : Model(1, 1, 1e-4, 1e-6, 0.1, false) {}

  /* observe(): initialize member data given an input data set.
   *
   * arguments:
   *  @xa: [m, k], target alpha points.
   *  @xb: [m, k], target beta points.
   *  @ya: [n, k], source alpha points.
   *  @yb: [n, k], source beta points.
   *  @prior_matching: [m, n], optional value of @omega0.
   *  @initial_rotation: [k, k], optional initial rotation matrix.
   */
  void observe(
    const Eigen::Ref<const Eigen::MatrixXd> xa,
    const Eigen::Ref<const Eigen::MatrixXd> xb,
    const Eigen::Ref<const Eigen::MatrixXd> ya,
    const Eigen::Ref<const Eigen::MatrixXd> yb,
    std::optional<Eigen::MatrixXd> prior_matching = {},
    std::optional<Eigen::MatrixXd> initial_rotation = {}
  ) {
    const auto m = xa.rows();
    const auto n = ya.rows();
    const auto k = xa.cols();

    if (xb.rows() != m)
      throw std::length_error("target row count mismatch");

    if (yb.rows() != n)
      throw std::length_error("source row count mismatch");

    if (xb.cols() != k || ya.cols() != k || yb.cols() != k)
      throw std::length_error("column count mismatch");

    if (k != 3)
      throw std::length_error("only three-dimensional data are supported");

    Eigen::MatrixXd R0 = Eigen::MatrixXd::Identity(k, k);
    if (initial_rotation) {
      R0 = *initial_rotation;
      if (R0.rows() != k || R0.cols() != k)
        throw std::length_error{"initial rotation size mismatch"};

      const auto I = Eigen::MatrixXd::Identity(k, k);
      if (!(R0.transpose() * R0).isApprox(I, 1.0e-6) || R0.determinant() <= 0)
        throw std::domain_error{"initial rotation is not a proper rotation"};
    }

    /* store the input data. */
    Xa = xa;
    Xb = xb;
    Ya = ya;
    Yb = yb;

    /* initialize the prior parameters. */
    omega0.resize(m, n);
    A0.resize(k, k);

    omega0.fill(1.0);
    if (prior_matching) {
      auto omega0_init = prior_matching.value();
      if (omega0_init.rows() != m || omega0_init.cols() != n)
        throw std::length_error("prior matching matrix size mismatch");

      if ((omega0_init.array() <= 0).any())
        throw std::domain_error("prior matching matrix must be positive");

      omega0 = omega0_init;
    }

    Eigen::VectorXd omega0_sums = omega0.rowwise().sum().cwiseInverse();
    omega0 = omega0_sums.asDiagonal() * omega0;
    ln_omega0 = omega0.array().log();

    { /* each point in each pair is uniform over the aabb of the target points.
       * every dimension of the bounding box is at least one unit long.
       */
      Eigen::MatrixXd X(2 * m, k);
      X << Xa, Xb;

      const Eigen::ArrayXd extent = X.colwise().maxCoeff() - X.colwise().minCoeff();
      ln_volume = extent.max(1.0).log().sum();
    }

    A0.setZero();
    A0.diagonal() = Eigen::ArrayXd::Constant(k, 1, rho0);

    ln_norm0 = matrix_fisher_lognorm(A0);
    ln_norm = ln_norm0;

    /* initialize the fit parameters. */
    eta.resize(k);
    A.resize(k, k);

    alpha = alpha0;
    beta = beta0;
    gamma = gamma0;
    eta.setZero();

    A = initial_rotation_concentration * R0;

    /* initialize the expectations from the priors. */
    t.resize(k);
    Qa.resize(m, n);
    Qb.resize(m, n);
    Z.resize(m, n + 1);
    R.resize(k, k);

    Z.leftCols(n) = (1 - w0) * omega0;
    Z.col(n).setConstant(w0);
    t.setZero();
    tt = k / gamma;

    /* update expectations and initialize the translation and precisions. */
    update_precision(/*modify=*/false);
    update_rotation(/*modify=*/false);

    update_translation(/*modify=*/true);
    update_precision(/*modify=*/true);
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
    update_assignments(modify);
    update_precision(modify);
    update_rotation(modify);
    update_translation(modify);

    return bound();
  }

  /* optimize(): optimize variational parameters.
   *
   * arguments:
   *  @iters: maximum number of optimization steps to perform.
   *  @th_abs: absolute threshold on the bound for early stopping.
   *  @th_rel: relative threshold on the bound for early stopping.
   *
   * returns:
   *  final value of the variational lower bound.
   */
  double optimize(
    unsigned int iters = 1000,
    double th_abs = 1.0e-6,
    double th_rel = 1.0e-9
  ) {
    double elbo = bound();

    for (unsigned int it = 0; it < iters; ++it) {
      const double elbo_prev = elbo;
      elbo = step();

      const double change = std::abs(elbo - elbo_prev);
      if (change < th_abs || change < th_rel * std::abs(elbo_prev))
        break;
    }

    return elbo;
  }

  /* bound(): compute the variational lower bound.
   */
  double bound() const {
    using std::numbers::pi;
    auto [m, n, k] = sizes();
    double elbo = 0;

    { /* E[ln p(D | Z, R, t, lambda)] */
      const Eigen::MatrixXd ll = (-0.5 * lambda * (Qa + Qb)).array()
                               + k * (ln_lambda - std::log(2 * pi));

      elbo += (matches().array() * ll.array()).sum();
    }

    { /* E[ln p(Z) - ln q(Z)] */
      const auto Zm = matches().array();
      const double base = std::log1p(-w0);
      elbo += (Zm > 0).select(
        /*true=*/Zm * (ln_omega0.array() - Zm.log() + base),
        /*false=*/0.0
      ).sum();
    }

    { /* unmatched point pair contributions. */
      if (w0 > 0) {
        const double base = std::log(w0) - 2 * ln_volume;
        elbo += (Z.col(n).array() > 0).select(
          /*true=*/Z.col(n).array() * (base - Z.col(n).array().log()),
          /*false=*/0.0
        ).sum();
      }
    }

    /* E[ln p(R) - ln q(R)] */
    elbo += (A0.array() * R.array()).sum() - ln_norm0;

    if (!use_argmax_rotation)
      elbo += ln_norm - (A.array() * R.array()).sum();

    /* E[ln p(t) - ln q(t)] */
    elbo += 0.5 * k * (std::log(gamma0) - std::log(gamma) + 1)
          - 0.5 * gamma0 * tt;

    /* E[ln p(lambda) - ln q(lambda)] */
    elbo += alpha0 * std::log(beta0) - std::lgamma(alpha0)
          + (alpha0 - 1) * ln_lambda - beta0 * lambda;

    elbo += alpha - std::log(beta) + std::lgamma(alpha)
          - (alpha - 1) * Eigen::numext::digamma(alpha);

    return elbo;
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
   *  @ln_omega0: [m, n], assignment log-weights.
   *  @A0: [k, k], rotation parameter.
   *  @ln_norm0: log-normalizer of the rotation prior.
   *  @w0:  prior probability that a target point pair is unmatched.
   *  @ln_volume: log-volume of the uniform assignment background density.
   *  @use_argmax_rotation: whether to approximate q(R) as a delta function.
   */
  double alpha0, beta0, gamma0, rho0;
  Eigen::MatrixXd omega0, ln_omega0;
  Eigen::MatrixXd A0;
  double ln_norm0, w0, ln_volume;
  bool use_argmax_rotation;

  /* parameters:
   *  @eta: [k], translation shift parameter.
   *  @A: [k, k], rotation parameter.
   *  @alpha: alignment noise shape.
   *  @beta: alignment noise rate.
   *  @gamma: translation precision.
   *  @ln_norm: log-normalizer of the rotation posterior approximation.
   */
  Eigen::VectorXd eta;
  Eigen::MatrixXd A;
  double alpha, beta, gamma, ln_norm;

  /* vector-valued expectations:
   *  @t: [k], translation first non-central moment.
   *
   * matrix-valued expectations:
   *  @Qa, @Qb: [m, n], squared alignment errors of alpha and beta.
   *  @Z: [m, n+1], assignment matrix.
   *  @R: [k, k], rotation matrix.
   *
   * scalar-valued expectations:
   *  @tt: translation inner product.
   *  @lambda: alignment noise precision.
   *  @ln_lambda: alignment noise log-precision.
   */
  Eigen::VectorXd t;
  Eigen::MatrixXd Qa, Qb, Z, R;
  double tt, lambda, ln_lambda;

private:
  /* @initial_rotation_concentration: diagonal of the initial A.
   */
  static constexpr double initial_rotation_concentration = 1.0e4;

  /* @besi0, @besi1: modified bessel functions.
   */
  static constexpr vbasa::BesselI0 besi0{};
  static constexpr vbasa::BesselI1 besi1{};

  /* update_rotation(): update q(R), the rotation factor.
   */
  void update_rotation(bool modify) {
    if (modify) {
      const Eigen::MatrixXd W = lambda * matches();
      const Eigen::MatrixXd Xa_t = Xa.rowwise() - t.transpose();
      const Eigen::MatrixXd Xb_t = Xb.rowwise() - t.transpose();

      A = A0 + Xa_t.transpose() * W * Ya + Xb_t.transpose() * W * Yb;
    }

    Eigen::MatrixXd U, V;
    Eigen::VectorXd s;
    proper_svd(A, U, s, V);

    if (use_argmax_rotation) {
      R = U * V.transpose();
    }
    else {
      Eigen::VectorXd d(s.size());
      matrix_fisher_moments(s(0), s(1), s(2), ln_norm, d);
      R = U * d.asDiagonal() * V.transpose();
    }

    recompute_alignment_errors();
  }

  /* update_translation(): update q(t), the translation factor.
   */
  void update_translation(bool modify) {
    const unsigned int k = dims();

    if (modify) {
      const Eigen::MatrixXd W = lambda * matches();
      const Eigen::VectorXd wi = W.rowwise().sum();
      const Eigen::VectorXd wj = W.colwise().sum().transpose();

      gamma = gamma0 + 2 * W.sum();
      eta = (Xa + Xb).transpose() * wi - R * ((Ya + Yb).transpose() * wj);
    }

    t = eta / gamma;
    tt = k / gamma + eta.dot(eta) / (gamma * gamma);

    recompute_alignment_errors();
  }

  /* update_assignments(): update q(Z), the assignments factor.
   */
  void update_assignments(bool modify) {
    if (!modify)
      return;

    const auto [m, n, k] = sizes();
    using std::numbers::pi;

    Eigen::MatrixXd ln_zeta(m, n + 1);
    ln_zeta.leftCols(n) = ln_omega0.array() + std::log1p(-w0);
    ln_zeta.leftCols(n) -= 0.5 * lambda * (Qa + Qb);
    ln_zeta.leftCols(n).array() += double(k) * (ln_lambda - std::log(2 * pi));

    ln_zeta.col(n).setConstant(
      w0 > 0 ? std::log(w0) - 2 * ln_volume
             : -std::numeric_limits<double>::infinity()
    );

    /* renormalize each row in the log domain. */
    const Eigen::VectorXd ln_zeta_max = ln_zeta.rowwise().maxCoeff();
    ln_zeta.colwise() -= ln_zeta_max;
    Z = ln_zeta.array().exp();

    /* handle vectorized exp() clamping. */
    if (w0 == 0)
      Z.col(n).setZero();

    Eigen::VectorXd z_sums = Z.rowwise().sum().cwiseInverse();
    Z = z_sums.asDiagonal() * Z;
  }

  /* update_precision(): update q(lambda), the noise precision factor.
   */
  void update_precision(bool modify) {
    if (modify) {
      const double k = dims();
      const double z = matches().sum();
      const double zqq = (matches().array() * (Qa + Qb).array()).sum();

      alpha = alpha0 + k * z;
      beta = beta0 + 0.5 * zqq;
    }

    lambda = alpha / beta;
    ln_lambda = Eigen::numext::digamma(alpha) - std::log(beta);
  }

  /* matches(): return the part of the assignment matrix corresponding to matches.
   */
  Eigen::Block<const Eigen::MatrixXd, -1, -1, true> matches() const {
    return Z.leftCols(num_source());
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
    const Eigen::VectorXd xs = X.rowwise().squaredNorm() - 2 * X * t;
    const Eigen::VectorXd ys = Y.rowwise().squaredNorm() + 2 * Y * (R.transpose() * t);

    Q.noalias() = -2 * X * (R * Y.transpose());
    Q.colwise() += xs;
    Q.rowwise() += ys.transpose();
    Q.array() += tt;
  }

  /* proper_svd(): compute a 'proper' singular value decomposition,
   *
   *   A = U diag(s) V'
   *
   * where U and V are rotations and only the final singular value
   * may be negative.
   */
  static void proper_svd(
    const Eigen::MatrixXd& A,
    Eigen::MatrixXd& U,
    Eigen::VectorXd& s,
    Eigen::MatrixXd& V
  ) {
    const auto options = Eigen::ComputeFullU | Eigen::ComputeFullV;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, options);

    U = svd.matrixU();
    V = svd.matrixV();
    s = svd.singularValues();

    const unsigned int k = A.rows();
    const double u_sign = U.determinant() < 0 ? -1.0 : 1.0;
    const double v_sign = V.determinant() < 0 ? -1.0 : 1.0;

    U.col(k - 1) *= u_sign;
    V.col(k - 1) *= v_sign;
    s(k - 1) *= u_sign * v_sign;
  }

  /* matrix_fisher_moments(): use gauss-legendre quadrature to compute the
   * normalizing constant and expected singular values of a matrix
   * fisher density.
   */
  static void matrix_fisher_moments(
    double si,
    double sj,
    double sk,
    double& ln_C,
    Eigen::Ref<Eigen::VectorXd> es
  ) {
    static const vbasa::GaussLegendre</*n=*/16> gl;

    const double a = 0.5 * (si - sj);
    const double b = 0.5 * (si + sj);
    const double c = sj + sk;

    /* g0: c(s) * exp(-(s1 + s2 + s3)).
     * g1, g2, g3: the derivatives of c(s) by s1, s2, s3, scaled alike.
     */
    double g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    for (int p = 0; p + 1 < gl.n_edges; ++p) {
      const double mid = 0.5 * (gl.edges[p + 1] + gl.edges[p]);
      const double half = 0.5 * (gl.edges[p + 1] - gl.edges[p]);

      for (int q = 0; q < gl.n_nodes; q++) {
        const double v = mid + half * gl.x[q];
        const double sm = a * v;
        const double sp = b * (2 - v);

        const double i0m = besi0.scaled(sm);
        const double i1m = besi1.scaled(sm);
        const double i0p = besi0.scaled(sp);
        const double i1p = besi1.scaled(sp);
        const double e = half * gl.w[q] * std::exp(-c * v);

        g0 += 0.5 * i0m * i0p * e;
        g1 += 0.25 * (v * i1m * i0p + (2 - v) * i0m * i1p) * e;
        g2 += 0.25 * ((2 - v) * i0m * i1p - v * i1m * i0p) * e;
        g3 += 0.5 * (1 - v) * i0m * i0p * e;
      }
    }

    ln_C = si + sj + sk + std::log(g0);
    es(0) = g1 / g0;
    es(1) = g2 / g0;
    es(2) = g3 / g0;
  }

  /* matrix_fisher_lognorm(): compute the log-normalizer of the matrix Fisher.
   */
  static double matrix_fisher_lognorm(const Eigen::MatrixXd& A0) {
    const unsigned int k = A0.rows();

    Eigen::MatrixXd U, V;
    Eigen::VectorXd s, es(k);
    double norm;

    proper_svd(A0, U, s, V);
    matrix_fisher_moments(s(0), s(1), s(2), norm, es);
    return norm;
  }
};

} /* namespace vbasa */

