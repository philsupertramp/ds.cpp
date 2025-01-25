#pragma once

#include "../Matrix.h"
#include "Classifier.h"
#include "SGD.h"

/**
 * Adaline neuron classifier using gradient decent method
 */
class AdalineGD : public ANNClassifier
{
public:
  /**
   * default constructor
   * @param _eta learning rate
   * @param iter max number iterations
   */
  explicit AdalineGD(double _eta = 0.01, int iter = 10)
    : ANNClassifier(_eta, iter) { }

  /**
   * fit method to train the classifier using gradient decent method
   * @param X input
   * @param y target output
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    initialize_weights(X.columns());
    costs = Matrix<double>(0, n_iter, 1);
    for(int iter = 0; iter < n_iter; iter++) {
      auto output = netInput(X);
      auto errors = y - output;

      auto delta_w = (X.Transpose() * errors) * eta;

      update_weights(Matrix<double>(eta * errors.sumElements(), weights.rows(), weights.columns()), delta_w);
      auto cost      = costFunction(errors);
      costs(iter, 0) = cost;
    }
  }

  /**
   * calculates
   * $$ X \cdot w + b
   * @param X input values
   * @returns
   */
  Matrix<double> netInput(const Matrix<double>& X) override { return SGD::netInput(X, weights); }

  /**
   * activates given input
   * @param X input
   * @returns activated input
   */
  Matrix<double> activation(const Matrix<double>& X) override { return netInput(X); }

  /**
   * predicts class of given input
   * @param X input values
   * @returns prediction
   */
  Matrix<double> predict(const Matrix<double>& X) override {
    std::function<bool(double)> condition = [](double x) { return bool(x >= 0.0); };
    return where(condition, activation(X), { { 1 } }, { { -1 } });
  }

  /**
   * calculates cost
   *
   * $$\frac{\sum X^2}{2}
   * @param X input values
   * @returns squared cost
   */
  double costFunction(const Matrix<double>& X) override { return HadamardMulti<double>(X, X).sumElements() / 2.0; }
};
/**
 * \example ds/TestAdalineGD.cpp
 * This is an example on how to use the AdalineGD class for binary classification.
 *
 */
