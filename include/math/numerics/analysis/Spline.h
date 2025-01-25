/**
 * @file Spline.h
 *
 * Requires:
 * \code
 * #include <math/numerics/analysis/Spline.h>
 * \endcode
 */
#pragma once

#include "../../Matrix.h"
#include "../../matrix_utils.h"
#include "../lin_alg/gaussSeidel.h"
#include "../utils.h"

/**
 * Spline implementation.
 *
 * Capable of 1D Natural-Cubic-Splines as well as 2/3D Interpolation using approximations.
 */
class Spline
{
  //! x-axis support values
  Matrix<double> XI;
  //! y-axis support values
  Matrix<double> YI;
  //! z-axis support values
  Matrix<double> ZI;
  //! t-Values for x
  Matrix<double> Tx;
  //! t-Values for y
  Matrix<double> Ty;
  //! flag to signalize that support values do not lie equidistant to each other
  bool isEquidistant = true;

public:
  /**
   * Default constructor
   * @param X x-axis support values
   * @param Y y-axis support values
   */
  Spline(const Matrix<double>& X, const Matrix<double>& Y) {
    bool isRows = X.rows() > X.columns();
    XI          = isRows ? X : X.Transpose();
    YI          = isRows ? Y : Y.Transpose();
    double h    = fabs(XI(1, 0) - XI(0, 0));
    for(size_t i = 0; i < XI.rows() - 1; ++i) {
      auto dist = fabs(XI(i + 1, 0) - XI(i, 0));
      if((dist - h) > EPS) { isEquidistant = false; }
    }
  }

  /**
   * 3D approximation using a b=2 b-spline
   * @param X x-axis evaluation values
   * @param Y y-axis evaluation values
   * @param Z z-axis evaluation values
   */
  Spline(const Matrix<double>& X, const Matrix<double>& Y, const Matrix<double>& Z) {
    bool isRows = X.rows() > X.columns();
    XI          = isRows ? X : X.Transpose();
    YI          = isRows ? Y : Y.Transpose();
    ZI          = isRows ? Z : Z.Transpose();
    double h    = fabs(XI(1, 0) - XI(0, 0));
    for(size_t i = 0; i < XI.rows() - 1; ++i) {
      auto dist = fabs(XI(i + 1, 0) - XI(i, 0));
      if((dist - h) > EPS) { isEquidistant = false; }
    }
  }
  /**
   * Evaluation of cubic polynomial s_j
   * @returns
   */
  double eval_spline_j(double x_act, size_t j, const Matrix<double>& mi) {
    auto h = XI(j + 1, 0) - XI(j, 0);
    return ((mi(j, 0) * pow((XI(j + 1, 0) - x_act), 3.0) + mi(j + 1, 0) * pow((x_act - XI(j, 0)), 3.0)) / 6.0) / h
           + (YI(j, 0) * (XI(j + 1, 0) - x_act) + YI(j + 1, 0) * (x_act - XI(j, 0))) / h
           - ((mi(j, 0) * (XI(j + 1, 0) - x_act) + mi(j + 1, 0) * (x_act - XI(j, 0))) * h) / 6.0;
  }

  /**
   * Calculates second derivative on given knot-points
   * @param h distance between points
   * @returns
   */
  Matrix<double> curv(double h) {
    auto mi  = zeros(XI.rows(), XI.columns());
    auto dim = XI.rows() - 2;
    auto rhs = 6.0 / (h * h) * (tridiag(dim, dim, 1, -2, 1) * YI.GetSlice(1, YI.rows() - 2, 0, YI.columns() - 1));
    auto lhs = tridiag(dim, dim, 1, 4, 1);
    auto res = gaussSeidel(lhs, rhs);
    for(size_t i = 0; i < res.rows(); ++i) { mi(i + 1, 0) = res(i, 0); }
    return mi;
  }

  /**
   * Setter for abstraction values t
   * @param tx x-axis
   * @param ty y-axis
   */
  void SetAbstractionValue(const Matrix<double>& tx, const Matrix<double>& ty) {
    Tx = tx;
    Ty = ty;
  }

  /**
   * Evaluates the spline for given values
   * @param xi input values
   * @returns evaluated input
   */
  Matrix<double> operator()(const Matrix<double>& xi) {
    if(isEquidistant && ZI.rows() == 0) return calculateEquidistant(xi);
    else {
      Matrix<double> ti;
      if(Tx.rows() > 0) {
        ti = Tx;
      } else {
        ti = linspace(min(xi), max(xi), XI.rows()).Transpose();
      }
      auto s = Spline(ti, XI)(xi);

      if(YI.rows() > 0) { s = s.HorizontalConcat(Spline(ti, YI)(xi)); }
      if(ZI.rows() > 0) { s = s.HorizontalConcat(Spline(ti, ZI)(xi)); }
      return s;
    }
  }

  /**
   * Calculates natural cubic spline for given input xi
   * @param xi input values
   * @returns
   */
  Matrix<double> calculateEquidistant(const Matrix<double>& xi) {
    auto innerXI = xi.rows() > xi.columns() ? xi : xi.Transpose();
    auto mi      = curv(XI(1, 0) - XI(0, 0));
    auto y       = zeros(innerXI.rows(), innerXI.columns());
    // Elementwise evaluate splines, for all
    // elements xi with i = 0, ..., n-1
    for(size_t j = 0; j < XI.rows() - 1; ++j) {
      // all elements x, between x_j and x_{j+1}
      auto xl  = XI(j, 0);
      auto xr  = XI(j + 1, 0);
      auto ind = nonzero([xl, xr](const double& x) { return bool((xl <= x) && (x <= xr)); }, innerXI).Transpose();
      // evaluate using the j-th spline
      for(size_t i = 0; i < ind.rows(); ++i) { y(ind(i, 0), 0) += eval_spline_j(innerXI(ind(i, 0), 0), j, mi); }
    }
    return y;
  }
};

/**
 * \example numerics/analysis/TestSpline.cpp
 * This is an example on how to use the Spline class.
 */
