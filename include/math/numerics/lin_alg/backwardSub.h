/**
 * @file
 *
 * Regular backward substitution.
 * Calculates result given upper right triangular matrix
 * $$
 * z=R\cdot b \Rightarrow R^{-1}\cdot b = z
 * $$
 *
 * Requires:
 * \code
 * #include <math/numerics/lin_alg/backwardSub.h>
 * \endcode
 */

#pragma once
#include "../utils.h"
#include <iostream>
#include <vector>

/**
 * Backward-substitution
 * @param R Coefficient matrix
 * @param b result vector
 * @returns z
 */
Matrix<double> backwardSub(const Matrix<double>& R, const Matrix<double>& b) {
  size_t m  = R.rows();
  size_t n  = R.columns();
  size_t v  = b.rows();
  size_t nv = b.columns();
  if(v != m) {
    // Error, cannot compute
    std::cout << "Matrix vector dimension miss match, error!\n";
    return Matrix<double>();
  }
  if(m != n) {
    // Error, cannot compute
    std::cout << "Matrix not squared. Dimension miss match, error!\n";
    return Matrix<double>();
  }

  auto x = zeros(v, nv);

  if(R(m - 1, n - 1) != 0) { x.SetRow(v - 1, b(v - 1) * (1 / R(m - 1, n - 1))); }

  for(int j = (int)m - 2; j >= 0; j--) {
    double s_k = 0.0f;
    for(size_t k = j + 1; k < m; k++) { s_k += (R(j, k) * x(k, 0)); }
    for(size_t i = 0; i < nv; ++i) { x(j, i) = (b(j, i) - s_k) / R(j, j); }
  }

  return x;
}
/**
 * \example numerics/lin_alg/TestBackwardSub.cpp
 * This is an example on how to use backwardSub.
 */