#pragma once

#include "../Matrix.h"
#include "../matrix_utils.h"
#include "Classifier.h"
#include "SGD.h"

/**
 * Represents Perceptron definition by Frank Rosenblatt
 *
 * [Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
 *
 * Essentially it can be used to build binary classification
 * One might for example want to replace some very complex binary operation
 * which is a combination of several broad operations like OR|AND|XOR|NOT ...
 *
 * In order to use the perceptron classifier one needs access to the full set
 * of possible transformations.
 *
 * THIS IS NOT A PREDICTOR
 *
 * Mainly implemented for academic use
 */
class Perceptron : public ANNClassifier
{
public:
  /**
   * default constructor
   * @param _eta
   * @param iter
   */
  explicit Perceptron(double _eta = 0.01, int iter = 10)
    : ANNClassifier(_eta, iter) { }

  /**
   *
   * @param X: array-like with the shape: [n_samples, n_features]
   * @param y: array-like with shape: [n_samples, 1]
   * @returns this
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    initialize_weights(X.columns());
    costs = Matrix<double>(0, n_iter, 1);
    for(int iter = 0; iter < n_iter; iter++) {
      int _errors   = 0;
      auto iterable = zip(X, y);
      for(const auto& elem : iterable) {
        auto xi     = elem.first;
        auto target = elem.second;

        auto output  = predict(xi);
        auto delta_w = (target - output);
        update_weights(delta_w, (delta_w * xi).Transpose() * eta);
        _errors += costFunction(delta_w);
      }
      costs(iter, 0) = _errors;
    }
  }

  /**
   * calculates net input
   * \f[
   *      X * weights + b
   * \f]
   * @param X
   * @returns
   */
  Matrix<double> netInput(const Matrix<double>& X) override { return SGD::netInput(X, weights); }

  /**
   * activate given input
   * @param X
   * @returns
   */
  Matrix<double> activation(const Matrix<double>& X) override { return netInput(X); }

  /**
   * predict class of given input
   * @param X
   * @returns
   */
  Matrix<double> predict(const Matrix<double>& X) override {
    std::function<bool(double)> condition = [](double x) { return bool(x >= 0.0); };
    return where(condition, activation(X), { { 1 } }, { { -1 } });
  }

  /**
   * cost function override, don't use!
   * @param X
   * @returns
   */
  double costFunction(const Matrix<double>& X) override { return (double)(X(0, 0) != 0.0); }
};

/**
 * \example ds/TestPerceptron.cpp
 * This is an example on how to use the Perceptron class for binary classification.
 *
 */
