
#pragma once

namespace vbasa {

/* poly(): evaluate a polynomial.
 *
 * arguments:
 *  @coeffs: array of polynomial coefficients.
 *  @x: value at which to evaluate the polynomial.
 *
 * returns:
 *  polynomial value.
 */
template<unsigned int n>
double poly(const double (&coeffs)[n], double x) {
  double sum = coeffs[0];
  double xn = 1.0;

  for (unsigned int i = 1; i < n; i++) {
    xn *= x;
    sum += coeffs[i] * xn;
  }

  return sum;
}

/* BesselI0: zero-order modified bessel function.
 */
class BesselI0 {
public:
  /* operator()(): evaluate I0(x). */
  double operator()(double x) const {
    const double ax = std::abs(x);
    double P = 0, Q = 0;

    if (ax < 15) {
      const double y = x * x;
      P = poly(p_sm, y);
      Q = poly(q_sm, 225 - y);
    }
    else {
      const double z = 1 - 15 / ax;
      P = std::exp(ax) * poly(p_lg, z);
      Q = std::sqrt(ax) * poly(q_lg, z);
    }

    return std::abs(P / Q);
  }

private:
  static constexpr double p_sm[14] = {
    9.999999999999997e-1,
    2.466405579426905e-1,
    1.478980363444585e-2,
    3.826993559940360e-4,
    5.395676869878828e-6,
    4.700912200921704e-8,
    2.733894920915608e-10,
    1.115830108455192e-12,
    3.301093025084127e-15,
    7.209167098020555e-18,
    1.166898488777214e-20,
    1.378948246502109e-23,
    1.124884061857506e-26,
    5.498556929587117e-30
  };

  static constexpr double q_sm[5] = {
    4.463598170691436e-1,
    1.702205745042606e-3,
    2.792125684538934e-6,
    2.369902034785866e-9,
    8.965900179621208e-13
  };

  static constexpr double p_lg[5] = {
    1.192273748120670e-1,
    1.947452015979746e-1,
    7.629241821600588e-2,
    8.474903580801549e-3,
    2.023821945835647e-4
  };

  static constexpr double q_lg[6] = {
    2.962898424533095e-1,
    4.866115913196384e-1,
    1.938352806477617e-1,
    2.261671093400046e-2,
    6.450448095075585e-4,
    1.529835782400450e-6
  };
};

/* BesselI1: first-order modified bessel function.
 */
class BesselI1 {
public:
  /* operator()(): evaluate I1(x). */
  double operator()(double x) const {
    const double ax = std::abs(x);
    double P = 0, Q = 0;

    if (ax < 15) {
      const double y = x * x;
      P = x * poly(p_sm, y);
      Q = poly(q_sm, 225 - y);
    }
    else {
      const double z = 1 - 15 / ax;
      P = std::exp(ax) * poly(p_lg, z);
      Q = std::sqrt(ax) * poly(q_lg, z);
    }

    return std::abs(P / Q);
  }

private:
  static constexpr double p_sm[14] = {
    5.000000000000000e-1,
    6.090824836578078e-2,
    2.407288574545340e-3,
    4.622311145544158e-5,
    5.161743818147913e-7,
    3.712362374847555e-9,
    1.833983433811517e-11,
    6.493125133990706e-14,
    1.693074927497696e-16,
    3.299609473102338e-19,
    4.813071975603122e-22,
    5.164275442089090e-25,
    3.846870021788629e-28,
    1.712948291408736e-31
  };

  static constexpr double q_sm[5] = {
    4.665973211630446e-1,
    1.677754477613006e-3,
    2.583049634689725e-6,
    2.045930934253556e-9,
    7.166133240195285e-13
  };

  static constexpr double p_lg[5] = {
    1.286515211317124e-1,
    1.930915272916783e-1,
    6.965689298161343e-2,
    7.345978783504595e-3,
    1.963602129240502e-4
  };

  static constexpr double q_lg[6] = {
    3.309385098860755e-1,
    4.878218424097628e-1,
    1.663088501568696e-1,
    1.473541892809522e-2,
    1.964131438571051e-4,
   -1.034524660214173e-6
  };
};

} /* namespace vbasa */

