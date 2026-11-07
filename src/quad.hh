
#pragma once

namespace vbasa {

/* GaussLegendre: nodes and weights of an n-point gauss-legendre quadrature rule.
 */
template<unsigned int n>
class GaussLegendre {
public:
  /* number of edges used in quadrature. */
  static constexpr int levels = 40;
  static constexpr int n_edges = 2 * levels + 3;
  static constexpr int n_nodes = n;

  /* @x: nodes.
   * @w: weights.
   * @edges: edges set.
   */
  double x[n], w[n];
  double edges[n_edges];

  /* GaussLegendre(): empty constructor. */
  GaussLegendre() {
    /* initialize nodes and weights. */
    using std::numbers::pi;
    for (unsigned int i = 0; i < n; ++i) {
      double z = std::cos(pi * (i + 0.75) / (n + 0.5));
      double dp = 1;

      for (unsigned int it = 0; it < 100; ++it) {
        double p1 = 1, p2 = 0;
        for (unsigned int j = 0; j < n; ++j) {
          const double p2_prev = p2;
          p2 = p1;
          p1 = ((2 * j + 1) * z * p2 - j * p2_prev) / (j + 1);
        }

        dp = n * (z * p1 - p2) / (z * z - 1);
        const double dz = p1 / dp;
        z -= dz;

        if (std::abs(dz) <= 2 * std::numeric_limits<double>::epsilon())
          break;
      }

      x[i] = z;
      w[i] = 2 / ((1 - z * z) * dp * dp);
    }

    /* initialize edges. */
    unsigned int idx = 0;
    edges[idx++] = 0;
    for (int lev = levels; lev >= 1; --lev)
      edges[idx++] = std::ldexp(1.0, -lev);

    edges[idx++] = 1;
    for (int lev = 1; lev <= levels; ++lev)
      edges[idx++] = 2 - std::ldexp(1.0, -lev);

    edges[idx++] = 2;
  }
};

} /* namespace vbasa */

