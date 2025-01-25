/**
 * @file ODESolver.h
 *
 *
 * Requires:
 * \code
 * #include <math/numerics/ode/ODESolver.h>
 * \endcode
 */
#pragma once

#include "../../Matrix.h"
#include "ExplicitEuler.h"
#include "ode.h"
#include "ode45.h"
#include "odeBDF2.h"
#include "odeTrapez.h"

/**
 * Proxy interface class for ODE solvers with static member functions as bridge
 * to solvers.
 */
class ODESolver
{
public:
  /**
   * proxy to ODEExpEuler
   * @param fun ode to approximate
   * @param tInterval interval to perform approximation on
   * @param y0 start value
   * @param option solver options
   * @returns ::ODEExpEuler(fun, tInterval, y0, option.h)
   */
  static ODEResult
  odeExpEuler(const ODE& fun, const std::vector<double>& tInterval, const Matrix<double>& y0, const ODEOption& option) {
    return ODEExpEuler(fun, tInterval, y0, option.h);
  }
  /**
   * proxy to ODE45
   * @param fun ode to approximate
   * @param tInterval interval to perform approximation on
   * @param y0 start value
   * @param option solver options
   * @returns ::ODE45(fun, tInterval, y0, option.h)
   */
  static ODEResult
  ode45(const ODE& fun, const std::vector<double>& tInterval, const Matrix<double>& y0, const ODEOption& option) {
    return ODE45(fun, tInterval, y0, option.h);
  }
  /**
   * proxy to ODETrapez
   * @param fun ode to approximate
   * @param tInterval interval to perform approximation on
   * @param y0 start value
   * @param option solver options
   * @returns ODETrapez(fun, tInterval, y0, option)
   */
  static ODEResult
  odeTrapez(const ODE& fun, const std::vector<double>& tInterval, const Matrix<double>& y0, const ODEOption& option) {
    return ODETrapez(fun, tInterval, y0, option);
  }
  /**
   * proxy to ODEBDF2
   * @param fun ode to approximate
   * @param tInterval interval to perform approximation on
   * @param y0 start value
   * @param option solver options
   * @returns ODEBDF2(fun, tInterval, y0, option)
   */
  static ODEResult
  odeBDF2(const ODE& fun, const std::vector<double>& tInterval, const Matrix<double>& y0, const ODEOption& option) {
    return ODEBDF2(fun, tInterval, y0, option);
  }
};