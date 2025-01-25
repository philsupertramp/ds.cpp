#pragma once

#include "../Matrix.h"
#include "Predictor.h"


/**
 * Classifier template class
 *
 * Inherit and implement:
 * - fit: method used to train the classifier
 *      - initializes weights
 *      - cycles through epochs
 *          - updates weights
 *          - calculates cost of current epoch
 * - predict: method used to predict the output for given input
 * - activation: method to activate a given input
 * - netInput: converts activated output into net output
 * - costFunction: method to calculate the current cost
 */
class Classifier : public Predictor
{
protected:
  //! flag to initialize weights only once
  bool w_initialized = false;

public:
  //! Vector holding weights
  Matrix<double> weights;
  //! Vector holding classification error per epoch
  Matrix<double> costs;

  /**
   * default constructor
   * @param _eta learning rate
   * @param _n_iter number learning iterations
   */
  Classifier() { }

  /**
   * Initialization method for weights hold by the classifier
   * @param numRows number rows
   * @param numColumns number columns
   */
  void initialize_weights(size_t numRows, size_t numColumns = 1) {
    weights       = Matrix<double>(0, numRows + 1, numColumns);
    w_initialized = true;
  }

  void update_weights(const Matrix<double>& update, const Matrix<double>& delta) {
    for(size_t i = 0; i < weights.rows(); i++) {
      for(size_t j = 0; j < weights.columns(); j++) {
        if(i == 0) weights(i, j) += update(i, j);
        else
          weights(i, j) += delta(i - 1, j);
      }
    }
  }
  Matrix<double> transform(const Matrix<double>& in) override { return in; }
};


/**
 * Artificial Neural Network Classifier interface.
 *
 * Adds interface for activation, cost and net-input functions.
 * As well as learning rate and constant for number of training epochs.
 */
class ANNClassifier : public Classifier
{
protected:
  //! Learning rate
  double eta;
  //! number epochs
  int n_iter;

public:
  ANNClassifier(double _eta, int _n_iter)
    : Classifier()
    , eta(_eta)
    , n_iter(_n_iter) { }

  /**
   * Activates a given input
   * @returns
   */
  virtual Matrix<double> activation(const Matrix<double>&) = 0;

  /**
   * calculates the net-input (aka output) this is mostly the derivative of `activation`
   * @returns
   */
  virtual Matrix<double> netInput(const Matrix<double>&) = 0;

  /**
   * calculates the current cost
   * if unused override returning 0
   */
  virtual double costFunction(const Matrix<double>&) = 0;
};
