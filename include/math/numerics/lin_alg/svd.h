#pragma once

#include "../../Matrix.h"
#include "gaussJordan.h"
#include "qr.h"

/**
 * Calculates SVD (Singular Value Decomposition) of given Matrix A for k Dimensions
 * @param A Matrix to operate with
 * @param k dimensions to calculate. If = 0 calculate all
 * @param epsilon squared error tolerance
 * @returns SVD of A
 */
std::vector<Matrix<double>> svd(const Matrix<double>& A, const size_t& k, const double epsilon = 0.1e-4) {
  // works with squared and not squared matrizes
  size_t n_orig   = A.rows();
  size_t m_orig   = A.columns();
  size_t chosen_k = k;
  if(k == 0) { chosen_k = n_orig < m_orig ? n_orig : m_orig; }
  auto A_copy = A;
  auto A_orig = A;
  size_t n, m;
  if(n_orig > m_orig) {
    A_copy = A_copy.Transpose() * A_copy;
    n      = A_copy.rows();
    m      = A_copy.columns();
  } else if(n_orig < m_orig) {
    A_copy = A_copy * A_copy.Transpose();
    n      = A_copy.rows();
    m      = A_copy.columns();
  } else {
    n = n_orig;
    m = m_orig;
  }

  auto Q      = Matrix<double>::Random(n, chosen_k);
  auto R      = Matrix<double>::Random(n, chosen_k);
  auto res    = qr(Q);
  Q           = res.first;
  auto Q_prev = Q;

  for(size_t i = 0; i < 1000; i++) {
    auto Z    = A_copy * Q;
    auto _res = qr(Z);
    Q         = _res.first;
    R         = _res.second;
    auto err  = ((Q - Q_prev)).Apply([](float val) { return val * val; }).sumElements();
    Q_prev    = Q;
    if(err < epsilon) { break; }
  }
  auto singular_values = diag_elements(R).Apply([](double val) { return sqrt(val); });
  Matrix<double> left_vecs, right_vecs;
  if(n_orig < m_orig) {
    left_vecs  = Q.Transpose();
    right_vecs = (gaussJordan(diag(singular_values)) * left_vecs.Transpose()) * A;
  } else if(n_orig == m_orig) {
    left_vecs       = Q.Transpose();
    right_vecs      = left_vecs;
    singular_values = singular_values.Apply([](double val) { return val * val; });
  } else {
    right_vecs = Q.Transpose();
    left_vecs  = A * (right_vecs.Transpose() * gaussJordan(diag(singular_values)));
  }

  return { left_vecs, singular_values, right_vecs };
}

/**
 * \example numerics/lin_alg/TestSVD.cpp
 * This is an example on how to use the svd method.
 */
