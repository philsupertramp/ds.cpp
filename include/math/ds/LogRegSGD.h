#pragma once

#include "../Matrix.h"
#include "Classifier.h"
#include "SGD.h"
#include "utils.h"

/**
 * Logistical regression model using statistic gradient decent method for model fitting.
 */
class LogRegSGD : public ANNClassifier
{
public:
  //! signalizes whether given dataset should be shuffled while fitting
  bool shuffle;
  //! initialize weights with random state
  int randomState;
  //! algorithmic object to represent fitting algorithm
  SGD* sgd = nullptr;

private:
  /**
   * helper to initialize weights matrix
   * @param m dimension
   */
  void initialize_weights(size_t m) {
    if(randomState) weights = Matrix<double>::Random(m, 1);
    else
      weights = Matrix<double>(0, m, 1);
    w_initialized = true;
  }


  /**
   * logistical regression
   * @param xi input values
   * @param target  target output values
   * @returns cost
   */
  double update_weights(const Matrix<double>& xi, const Matrix<double>& target) {
    auto output      = netInput(xi);
    auto error       = target - output;
    auto class1_cost = (target * -1.0) * Log(output);
    auto class2_cost = (Matrix<double>(1, output.rows(), output.columns()) - target)
                       * Log(Matrix<double>(1, output.rows(), output.columns()) - output);
    auto cost = class1_cost - class2_cost;

    auto gradient = (xi.Transpose() * error) * eta;
    gradient      = gradient * (1.0 / xi.rows());

    this->weights = this->weights - gradient;

    return cost.sumElements() / xi.rows();
  }

  /**
   * Calculates log of each element.
   * Implementation doesn't use Matrix::Apply
   * @param in input matrix
   * @returns
   */
  static Matrix<double> Log(const Matrix<double>& in) {
    auto out = in;
    for(size_t i = 0; i < out.rows(); i++) {
      for(size_t j = 0; j < out.columns(); j++) { out(i, j) = log(out(i, j)); }
    }
    return out;
  }

public:
  /**
   * default constructor
   * @param _eta learning rate
   * @param iter number learning iterations
   * @param _shuffle shuffle data in each iteration?
   * @param _randomState seed for random state
   */
  explicit LogRegSGD(double _eta = 0.01, int iter = 10, bool _shuffle = false, int _randomState = 0)
    : ANNClassifier(_eta, iter)
    , shuffle(_shuffle)
    , randomState(_randomState) {
    if(randomState != 0) Random::SetSeed(randomState);
  }

  /**
   * fits the weights of the model for given input values
   * @param X: array-like with the shape: [n_samples, n_features]
   * @param y: array-like with shape: [n_samples, 1]
   * @returns this
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    if(sgd == nullptr) {
      sgd = new SGD(
      eta,
      n_iter,
      shuffle,
      [this](const Matrix<double>& x, const Matrix<double>& y) { return this->update_weights(x, y); },
      [this](const Matrix<double>& x) { return this->netInput(x); });
    }
    initialize_weights(X.columns());
    sgd->fit(X, y, weights);
  }


  /**
   * partially fit weights
   * @param X input values
   * @param y target output values
   */
  void partial_fit(const Matrix<double>& X, const Matrix<double>& y) {
    if(!w_initialized) { initialize_weights(X.columns()); }
    sgd->partial_fit(X, y, weights);
  }

  /**
   * calculate netinput
   * @param X input values
   * @returns
   */
  Matrix<double> netInput(const Matrix<double>& X) override { return Sigmoid(X * weights); }

  /**
   * activate given input
   * $$x\mapsto \phi(x)$$
   * @param X input values
   * @returns activated values
   */
  Matrix<double> activation(const Matrix<double>& X) override { return netInput(X); }

  /**
   * predict output for given input
   * @param X input values
   * @returns predicted output for given input
   */
  Matrix<double> predict(const Matrix<double>& X) override {
    std::function<bool(double)> condition = [](double x) { return bool(x >= EPS); };
    return where(condition, activation(X), { { 1 } }, { { -1 } });
    //        return activation(X);
  }

  /**
   * do not use!
   * @param mat
   * @returns
   */
  double costFunction([[maybe_unused]] const Matrix<double>& mat) override { return 0; }
};


/**
 * \example ds/TestLogRegSGD.cpp
 * This is an example on how to use the LogRegSGD class for regression classification.
 *
 */
