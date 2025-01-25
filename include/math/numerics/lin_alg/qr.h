#pragma once
#include "../../Matrix.h"
#include "../utils.h"


/**
 * Calculates QR decomposition of given matrix
 * so in = Q * R
 * @param in matrix to calculate with
 * @returns { Q, R } decomposition
 */
std::pair<Matrix<double>, Matrix<double>> qr(const Matrix<double>& A) {
  size_t num_rows = A.rows();
  size_t num_cols = A.columns();

  auto Q     = zeros(num_rows, num_rows);
  size_t col = 0;

  for(size_t i_a = 0; i_a < num_cols; i_a++) {
    auto a = A.GetSlice(0, num_rows - 1, i_a, i_a);
    auto u = a;
    for(size_t i = 0; i < col; i++) {
      auto q    = Q.GetSlice(0, num_rows - 1, i, i);
      auto proj = (q.Transpose() * a)(0, 0) * q;
      u -= proj;
    }
    auto e = 1.0 / norm(u) * u;
    Q.SetSlice(0, e.rows() - 1, col, col, e);
    col += 1;
  }
  return std::make_pair(Q, Q.Transpose() * A);
}

/**
 * \example numerics/lin_alg/TestQR.cpp
 * This is an example on how to use the qr decomposition.
 */
