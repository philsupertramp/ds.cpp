#pragma once

#include "../Matrix.h"
#include "../matrix_utils.h"
#include <functional>


/**
 * Implements statistic gradient decent method to fit a model
 */
class SGD
{
  //! number iterations (epochs) during fit
  size_t n_iter;
  //! cost factor
  double eta;
  //! shuffle during training
  bool shuffle = false;
  //! represents weight update function
  std::function<double(const Matrix<double>&, const Matrix<double>&)> weight_update = nullptr;
  //! represents net input function
  std::function<Matrix<double>(const Matrix<double>&)> net_input_fun = nullptr;

public:
  //! matrix holding cost per epoch
  Matrix<double> cost;

  /**
   * default constructor
   * @param _eta
   * @param iter
   * @param _shuffle
   * @param update_weights_fun
   * @param netInputFun
   */
  explicit SGD(
  double _eta                                                                                   = 0.01,
  size_t iter                                                                                   = 10,
  bool _shuffle                                                                                 = false,
  const std::function<double(const Matrix<double>&, const Matrix<double>&)>& update_weights_fun = nullptr,
  const std::function<Matrix<double>(const Matrix<double>&)>& netInputFun                       = nullptr)
    : n_iter(iter)
    , eta(_eta)
    , shuffle(_shuffle) {
    if(update_weights_fun != nullptr) { weight_update = update_weights_fun; }
    if(netInputFun != nullptr) { net_input_fun = netInputFun; }
    cost = Matrix<double>(0.0, 0, 0);
  }

  /**
   * fits given weights according to used weight update/net input function based on given
   * values X and y
   * @param X
   * @param y
   * @param weights
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y, Matrix<double>& weights) {
    cost       = Matrix<double>(0, n_iter, 1);
    auto xCopy = X;
    auto yCopy = y;
    for(size_t iter = 0; iter < n_iter; iter++) {
      if(shuffle) {
        auto fooPair = shuffleData(X, y);
        xCopy        = fooPair.first;
        yCopy        = fooPair.second;
      }
      double costSum = 0;
      for(const auto& elem : zip(xCopy, yCopy)) {
        if(weight_update != nullptr) {
          costSum += weight_update(elem.first, elem.second);
        } else {
          costSum += update_weights(elem.first, elem.second, weights);
        }
      }

      cost(iter, 0) = costSum;
    }
  }

  /**
   * Performs partial fit of given weights to input values
   * @param X
   * @param y
   * @param weights
   */
  void partial_fit(const Matrix<double>& X, const Matrix<double>& y, Matrix<double>& weights) const {
    if(y.rows() > 1) {
      for(const auto& elem : zip(X, y)) { update_weights(elem.first, elem.second, weights); }
    } else {
      update_weights(X, y, weights);
    }
  }

  /**
   * calculates update values (mean-square-error)
   * @param xi
   * @param target
   * @param weights
   * @returns
   */
  double update_weights(const Matrix<double>& xi, const Matrix<double>& target, Matrix<double>& weights) const {
    auto output = Matrix<double>();
    if(net_input_fun != nullptr) {
      output = net_input_fun(xi);
    } else {
      output = netInput(xi, weights);
    }
    auto error = target - output;
    weights.SetRow(0, weights(0) + eta * error);
    auto delta = eta * (xi.Transpose() * error);
    for(size_t i = 0; i < weights.rows() - 1; i++) { weights.SetRow(i + 1, weights(i + 1) + delta(i)); }
    return ((error * error) * 0.5).sumElements() / (double)target.rows();
  }

  /**
   * shuffles given data
   * @param X
   * @param y
   * @returns
   */
  std::pair<Matrix<double>, Matrix<double>>
  shuffleData([[maybe_unused]] const Matrix<double>& X, [[maybe_unused]] const Matrix<double>& y) {
    return { X, y };
  }

  /**
   * computes the net-input for given values
   * @param X
   * @param weights
   * @returns
   */
  [[nodiscard]] static Matrix<double> netInput(const Matrix<double>& X, const Matrix<double>& weights) {
    Matrix<double> A(0, weights.rows() - 1, weights.columns());
    Matrix<double> B(0, 1, weights.columns());
    for(size_t i = 0; i < weights.rows(); i++) {
      for(size_t j = 0; j < weights.columns(); j++) {
        if(i == 0) {
          B(i, j) = weights(i, j);
        } else {
          A(i - 1, j) = weights(i, j);
        }
      }
    }

    // Note: since Matrix<T> will never allow Matrix-Scalar Addition
    //       we need to create a vector of size of rows of the input
    //       values (jeez, horrible sentence). Therefore we rescale
    //       B and assign the bias to each element.
    B = Matrix<double>(B(0, 0), X.rows(), 1);
    return (X * A) + B;
  }
};
