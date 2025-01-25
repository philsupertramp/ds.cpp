/**
 * @file
 *
 *
 * Implements BDF(2)-Method to solve systems of ordinary differential equations
 *
 * Requires:
 * \code
 * #include <math/numerics/ode/odeBDF2.h>
 * \endcode
 */
#pragma once

#include "../lin_alg/newton.h"
#include "../utils.h"
#include "ode.h"
#include "odeTrapez.h"
#include <iostream>


/**
 * BDF(2) solver implementation
 * @param fun ode to approximate
 * @param tInterval interval to perform approximation on
 * @param y0 start value
 * @param option solver options
 * @returns approximated values
 */
ODEResult
ODEBDF2(const ODE& fun, const std::vector<double>& tInterval, const Matrix<double>& y0, const ODEOption& option) {
  size_t tDim      = tInterval.size();
  size_t elem_size = y0.columns();
  auto h           = option.h;
  auto TOL         = option.TOL;
  auto maxIter     = option.maxIter;
  auto Jac         = option.Jac;
  if(h == 0) { h = (tInterval[tDim - 1] - tInterval[0]) / 1000.0; }

  //    % initialize result vectors
  size_t result_dim = ((tInterval[tDim - 1] - tInterval[0]) / h) + 1;
  Matrix<double> t(0, result_dim, 1, 1);
  size_t n = result_dim;
  auto y   = zeros(n, elem_size);
  Matrix<int> iter(0, n, 1);

  t(0, 0) = tInterval[0];
  t(1, 0) = t(0, 0) + h;

  auto y1 = ODETrapez(fun, { t(0, 0), t(1, 0) }, y0, option).Y;
  y.SetRow(0, y1(0));
  y.SetRow(1, y1(1));
  auto E = eye(elem_size);

  for(size_t l = 1; l < n - 1; l++) {
    auto y_act = y(l - 1).Transpose();
    auto k     = 0;
    auto delta = ones(elem_size);

    while(norm(delta) > TOL && k < maxIter) {
      auto current_fun = fun(t(l, 0) + h, y_act);
      auto F =
      -1.0 * ((-1.0 / 2.0) * y(l - 1).Transpose() + h * current_fun + 2.0 * y(l).Transpose() - 3.0 / 2.0 * y_act);
      auto J = h * Jac(t(l, 0), y_act) - (3.0 / 2.0) * E;
      delta  = gaussSeidel(J, F);
      y_act += delta;
      k += 1;
    }
    if(k >= maxIter) { std::cout << "Warning: Max number iterations reached." << std::endl; }
    iter(l, 0)  = k;
    t(l + 1, 0) = t(l, 0) + h;
    y.SetRow(l + 1, y_act);
  }
  return { y, t, iter };
}

/**
 * \example numerics/ode/TestODEBDF2.cpp
 * This is an example on how to use ODEBDF2.
 */