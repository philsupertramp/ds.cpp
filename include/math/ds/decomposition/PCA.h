#pragma once

#include "../../numerics/lin_alg/svd.h"
#include "../Predictor.h"

class PCA : public Transformer
{
public:
  Matrix<double> PCs;
  Matrix<double> right;
  int keep_components;

public:
  PCA(int k_components = 0)
    : Transformer()
    , keep_components(k_components) { }

  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    auto res = svd(X, keep_components);
    PCs      = res[0] * res[1];
    right    = res[2];
  }
  Matrix<double> transform(const Matrix<double>& in) override { return in * PCs; }
};
