#pragma once

#include "../Matrix.h"
#include "Classifier.h"
#include "SGD.h"

/**
 * Adaline linear neuron implementation using statistic gradient decent
 * for training of a vector of weights and a single bias.
 *
 * weights: shape [N+1 x 1]
 * bias: shape [1, 1], first element of weights
 *
 * Note: only allows binary classification and uses a single layer of weights
 *       see the description here [1] I found it very easy to understand and
 *       implement.
 *
 *       - [1] http://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/
 */
class AdalineSGD : public ANNClassifier
{
public:
  //! signalizes whether given dataset should be shuffled while fitting
  bool shuffle;
  //! initialize weights with random state
  int randomState;
  //! algorithmic object to represent fitting algorithm
  SGD sgd;

public:
  /**
   * default constructor
   * @param _eta learning rate
   * @param iter number of learning iterations
   * @param _shuffle use shuffled data
   * @param _randomState seed of random state
   */
  explicit AdalineSGD(double _eta = 0.01, int iter = 10, bool _shuffle = false, int _randomState = 0)
    : ANNClassifier(_eta, iter)
    , shuffle(_shuffle)
    , randomState(_randomState) {
    if(randomState != 0) Random::SetSeed(randomState);
    sgd = SGD(_eta, iter, _shuffle);
  }

  /**
   * fit weights using sgd member
   * @param X: array-like with the shape: [n_samples, n_features]
   * @param y: array-like with shape: [n_samples, 1]
   * @returns this
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    initialize_weights(X.columns());
    sgd.fit(X, y, weights);
  }

  /**
   * partially fits the model
   * @param X input values
   * @param y target output values
   */
  void partial_fit(const Matrix<double>& X, const Matrix<double>& y) {
    if(!w_initialized) { initialize_weights(X.columns()); }
    sgd.partial_fit(X, y, weights);
  }

  /**
   * alias for sdg.netInput
   * @param X input values
   * @returns
   */
  Matrix<double> netInput(const Matrix<double>& X) override { return sgd.netInput(X, weights); }

  /**
   * Do not use
   * @param X
   * @returns
   */
  double costFunction([[maybe_unused]] const Matrix<double>& X) override { return 0; }

  /**
   * activates given input
   * @param X input values
   * @returns activated input values
   */
  Matrix<double> activation(const Matrix<double>& X) override { return netInput(X); }

  /**
   * predict output class of given input
   * @param X input values
   * @returns predicted output
   */
  Matrix<double> predict(const Matrix<double>& X) override {
    std::function<bool(double)> condition = [](double x) { return bool(x >= 0.0); };
    return where(condition, activation(X), { { 1 } }, { { -1 } });
  }
};


/**
 * \example ds/TestAdalineSGD.cpp
 * This is an example on how to use the AdalineSGD class for binary classification.
 *
 */
