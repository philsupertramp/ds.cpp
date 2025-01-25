/**
 * @file
 *
 * Gaussian elimination to solve systems of linear equations.
 * Uses column major pivot elements to reduce number of operations.
 *
 * Requires:
 * \code
 * #include <math/numerics/lin_alg/gausseidel.h>
 * \endcode
 */
#pragma once
#include "../../Matrix.h"
#include "../../matrix_utils.h"
#include "LU.h"
#include "backwardSub.h"
#include "forwardSub.h"
#include <vector>

/**
 * Gauss-Jordan algorithm to calculate inverse of given squared matrix.
 *
 *
 * @param A squared Coefficient matrix
 * @returns $$A^{-1}$$
 */
Matrix<double> gaussJordan(const Matrix<double>& A) {
  size_t i, j, k;
  size_t n = A.columns();
  double d;
  Matrix<double> mat = HorizontalConcat(A, eye(A.rows(), n));

  // Partial pivoting
  for(i = n - 1; i > 1; i--) {
    if(mat(i - 1, 1) < mat(i, 1)) {
      for(j = 0; j < (2 * n); ++j) {
        d             = mat(i, j);
        mat(i, j)     = mat(i - 1, j);
        mat(i - 1, j) = d;
      }
    }
  }

  // Reducing To Diagonal Matrix
  for(i = 0; i < n; i++) {
    for(j = 0; j < n; j++) {
      if(j != i) {
        d = mat(j, i) / mat(i, i);
        for(k = 0; k < n * 2; ++k) { mat(j, k) -= mat(i, k) * d; }
      }
    }
    d = mat(i, i);
    for(j = 0; j < (2 * n); j++) { mat(i, j) = mat(i, j) / d; }
  }

  return mat.GetSlice(0, mat.rows() - 1, n, mat.columns() - 1);
}

/**
 * \example numerics/lin_alg/TestGaussSeidel.cpp
 * This is an example on how to use gaussSeidel.
 */
