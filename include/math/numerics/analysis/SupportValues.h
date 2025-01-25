/**
 * function approximation and interpolation using
 * support values
 *
 * see https://wiki.godesteem.de/wiki/interpolation-and-approximation/
 *
 * Requires:
 * \code
 * #include <math/numerics/analysis/SupportValues.h>
 * \endcode
 */
#pragma once

#include "../../Matrix.h"
#include "../lin_alg/gaussSeidel.h"


/**
 * Template class for polynomial base representation.
 * - MonomBase
 * - LagrangeBase
 * - NewtonBase
 */
class PolynomialBase
{
public:
  /**
   * Default constructor
   * @param X support values $$x_i$$
   * @param Y evaluated support values $$y_i$$
   */
  PolynomialBase([[maybe_unused]] const Matrix<double>& X, [[maybe_unused]] const Matrix<double>& Y) { }

  /**
   * Evaluate a given vector based on the approximation,
   * computes p(x) for given x
   * @returns
   */
  virtual Matrix<double> Evaluate(const Matrix<double>&) const = 0;

  /**
   * string representation of interpolated polynomial
   *
   * Don't overthink this. simply return some kind of equation
   * one can copy into wolframalpha and simplify
   * @returns
   */
  virtual std::string Function() const = 0;

  /**
   * std::cout operator
   * @param ostr
   * @param base
   * @returns
   */
  friend std::ostream& operator<<(std::ostream& ostr, const PolynomialBase& base) {
    ostr << base.Function() << std::flush;
    return ostr;
  }
};

/**
 * Monom-Base:
 *
 * $$y_i = p(x_i) = a_0 + a_1x_i + ... + a_nx_i^n, \forall i = 0, \dots, n$$
 * https://wiki.godesteem.de/wiki/interpolation-and-approximation/#Monom-Base
 */
class MonomBase : public PolynomialBase
{
public:
  //! Coefficient matrix
  Matrix<double> A;

  /**
   * Default constructor
   * @param X $$x_i$$ support values
   * @param Y $$y_i$$ support values evaluated with function
   */
  MonomBase(const Matrix<double>& X, const Matrix<double>& Y)
    : PolynomialBase(X, Y) {
    auto x = X;
    // ensure required orientation
    if(x.columns() > x.rows()) { x = x.Transpose(); }
    Matrix<double> out(0, x.rows(), x.rows());
    for(size_t i = 0; i < out.rows(); ++i) {
      for(size_t j = 0; j < out.rows(); ++j) { out(i, j) = pow(x(i, 0), j); }
    }
    /**
     * quick test if everything is right,
     * if det(out) = 1 we made some mistake computing the vandermonde-matrix
     * per definition it's determinant is always \neq 0!
     */
    if(out.Determinant() == 0.0) {
      std::cerr << "Error computing Vandermonde-Matrix! Most likely passed equal support values" << std::endl;
      return;
    }
    A = gaussSeidel(out, Y);
  }
  /**
   * iterative evaluation, nothing special
   * @param X evaluation values
   * @returns approximation
   */
  virtual Matrix<double> Evaluate(const Matrix<double>& X) const {
    Matrix<double> Y(A(0, 0), X.rows(), 1);
    for(size_t i = 0; i < A.rows(); ++i) {
      Y += A(i, 0) * X.Apply([i](const double& in) { return pow(in, i); });
    }
    return Y;
  };
  /**
   * String representation getter
   * @returns string representation in monom base
   */
  virtual std::string Function() const {
    std::string out;
    for(size_t row = 0; row < A.rows(); ++row) {
      if(row != 0) { out += " + "; }
      out += format("%d", int(A(row, 0)));
      if(row > 0) { out += format("x^%d", row); }
    }
    return out;
  }
};

/**
 * Lagrange-Base using lagrange coefficients to build the polynomial
 *
 *
 * https://wiki.godesteem.de/wiki/interpolation-and-approximation/#Lagrange-Base
 */
class LagrangeBase : public PolynomialBase
{
  //! $$x_i$$ Support values
  Matrix<double> x;
  //! $$y_i$$ Support values evaluated with the function
  Matrix<double> y;

public:
  /**
   * Default constructor
   * @param X $$x_i$$
   * @param Y $$y_i$$
   */
  LagrangeBase(const Matrix<double>& X, const Matrix<double>& Y)
    : PolynomialBase(X, Y)
    , x(X)
    , y(Y) { }

  /**
   * Computes i-th coefficient
   *
   * $$L_i(x) = \prod_{j=0, j\neq i}^{n} \frac{x-x_j}{x_i-x_j}, \forall i=0, \dots, n$$
   * @param xk current value $$x_k$$
   * @param i index i
   * @returns coefficient value
   */
  [[nodiscard]] double GetCoefficient(const double& xk, const size_t i) const {
    double out = 1.0;
    for(size_t j = 0; j < x.rows(); ++j) {
      if(i == j) continue;

      out *= (xk - x(j, 0)) / (x(i, 0) - x(j, 0));
    }
    return out;
  }

  /**
   * Evaluates the lagrange base
   * @param in values to evaluate
   * @returns interpolated values
   */
  virtual Matrix<double> Evaluate(const Matrix<double>& in) const {
    Matrix<double> out(0, in.rows(), 1);
    for(size_t k = 0; k < in.rows(); ++k) {
      for(size_t i = 0; i < y.rows(); ++i) { out(k, 0) += y(i, 0) * GetCoefficient(in(k, 0), i); }
    }
    return out;
  }
  /**
   * builds lagrange coefficient $$L_i$$ string representation
   * @param i index
   * @returns coefficient representation
   */
  [[nodiscard]] std::string buildLx(size_t i) const {
    std::string Lx;

    bool needsOperator = false;
    for(size_t j = 0; j < x.rows(); ++j) {
      if(i == j) {
        needsOperator = false;
        continue;
      }

      // we need to exit, if one element of the product is 0
      if(x(i, 0) - x(j, 0) == 0) return "";

      if(needsOperator) { Lx += " * "; }
      if(x(i, 0) - x(j, 0) < 0) { Lx += "-"; }

      // Dividend
      if(x(j, 0) != 0) {
        if(x(j, 0) < 0) {
          Lx += format("(x + %.3f)", std::abs(x(j, 0)));
        } else {
          Lx += format("(x - %.3f)", x(j, 0));
        }
      } else {
        Lx += "x";
      }

      // Divisor
      Lx += format("/%.3f", std::abs(x(i, 0) - x(j, 0)));
      needsOperator = true;
    }
    return Lx;
  }
  /**
   * Builds string representation in lagrange base
   * @returns representational string
   */
  virtual std::string Function() const {
    std::string out;
    for(size_t i = 0; i < y.rows(); ++i) {
      if(y(i, 0) == 0) continue;

      if(i > 0) { out += " + "; }
      out += format("%d * (", int(y(i, 0)));
      out += buildLx(i);
      out += ")";
    }
    return out;
  }
};

/**
 * Newton-Base
 *
 *
 * $$
 * p(x) = \sum_{i=0}^{n} b_i \omega_i(x)
 * $$
 * with
 * $$
 *   b_i = \frac{f_{[x_{r+1}, ..., x_s]} - f_{[x_r, ..., x_{s-1}]}}{x_s - x_r}
 * $$
 * and
 * $$
 *   \omega_i(x) = \prod_{j=0}^{i-1} (x - x_j), \quad i = 1, \dots n
 * $$
 * and
 * $$
 * \omega_0(x) = x_0
 * $$
 *
 * https://wiki.godesteem.de/wiki/interpolation-and-approximation/#Newton-Base
 */
class NewtonBase : public PolynomialBase
{
  //! support values $$x_i$$
  Matrix<double> X;

public:
  //! newton base coefficients $$b_i$$
  Matrix<double> b;

  /**
   * default constructor
   * @param x support values $$x_i$$
   * @param y evaluated support values $$y_i$$
   */
  NewtonBase(const Matrix<double>& x, const Matrix<double>& y)
    : PolynomialBase(x, y)
    , X(x) {
    auto m = x.rows();
    Matrix<double> F(0, m, m);
    F.SetColumn(0, y);
    for(size_t j = 1; j < m; ++j) {
      for(size_t i = j; i < m; ++i) { F(i, j) = (F(i, j - 1) - F(i - 1, j - 1)) / (x(i, 0) - x(i - j, 0)); }
    }
    b = diag_elements(F);
  }
  /**
   * String representation in newton base
   * @returns representational string
   */
  virtual std::string Function() const {
    std::string out;
    for(size_t i = 0; i < b.rows(); ++i) {
      if(i > 0) { out += " + "; }
      out += format("%.3f", b(i, 0));
      if(i > 0) {
        for(size_t j = 0; j < i; ++j) {
          if(X(j, 0) != 0) {
            if(X(j, 0) > 0) out += format("* (x - %.3f)", X(j, 0));
            else
              out += format("* (x + %.3f)", std::abs(X(j, 0)));
          } else
            out += "* x";
        }
      }
    }
    return out;
  }

  /**
   * Getter for coefficient with index i for a value x
   * @param index index of coefficient
   * @param x value to evaluate
   * @returns evaluated coefficient
   */
  double GetCoefficient(size_t index, const double& x) const {
    double out = 1;
    for(size_t i = 0; i < index - 1; ++i) { out *= (x - X(i, 0)); }
    return out;
  }
  /**
   * Evaluate the newton base for given values
   * @param xIn values to evaluate
   * @returns interpolated values
   */
  virtual Matrix<double> Evaluate(const Matrix<double>& xIn) const {
    Matrix<double> y(b(b.rows() - 1, 0), xIn.rows(), 1);
    for(int j = b.rows() - 2; j >= 0; --j) {
      auto bAct = b(j, 0);
      auto xAct = X(j, 0);
      y         = xIn
          .Apply([xAct](const double& in) { return in - xAct; })  // (xIn - X[j])
          .HadamardMulti(y)                                       // (xIn - X[j]) * y
          .Apply([bAct](const double& in) { return bAct + in; }); // y = b[j] + (xIn - X[j]) * y
    }
    return y;
  }
};
/**
 * \example numerics/analysis/TestSupportValues.cpp
 * This is an example on how to use the MonomBase, LagrangeBase and NewtonBase class.
 */