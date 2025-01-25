/**
 * @file ode.h
 *
 * This header file holds utility helper for ode solvers
 *
 * Requires:
 * \code
 * #include <math/numerics/ode/ode.h>
 * \endcode
 */

#pragma once
#include <functional>
#include <vector>

//! alias for ODE
using ODE = std::function<Matrix<double>(double, Matrix<double>)>;

//! alias for Jacobian-Matrix of ODE
using ODEJac = std::function<Matrix<double>(double, Matrix<double>)>;

/**
 * Representation of ODE result
 */
struct ODEResult {
  //! Y-values
  Matrix<double> Y;
  //! T-values (mostly time)
  Matrix<double> T;
  //! number iterations per time step
  Matrix<int> Iterations;

  /**
   * Constructor without iterations
   * @param y y-values
   * @param t t-values
   */
  ODEResult(const Matrix<double>& y, const Matrix<double>& t)
    : Y(y)
    , T(t) { }

  /**
   * full element constructor
   * @param y y-values
   * @param t t-values
   * @param iter iteration-values
   */
  ODEResult(const Matrix<double>& y, const Matrix<double>& t, const Matrix<int>& iter)
    : Y(y)
    , T(t)
    , Iterations(iter) { }
};
/**
 * Option struct for ODE solvers
 */
struct ODEOption {
  //! step width of t
  double h = 0;
  //! tolerance for the algorithm to determine convergence
  double TOL = 1e-7;
  //! max iterations for integrated newton steps
  int maxIter = 50;
  //! jacobian matrix of given function
  ODEJac Jac = nullptr;
};