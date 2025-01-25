#pragma once
#include "../../Matrix.h"
#include "../../statistics/Probability.h"


/**
 * Standarize features by removing the mean and scaling to unit variance.
 *
 * The standard score of a sample $x\inX$ with $X\in\mathbf{R}^{N\times M}$ is calculated as
 * $$
 *  \tilde{x} = \frac{x - \mu}{\sigma}
 * $$
 * with
 * - $\mu\in\mathbf{R}^M$ the mean of the training samples
 * - $\sigma\in\mathbf{R}^M$ the standard deviation of the training samples
 *
 *
 * Note: Standardization of a data set is a common requirement for many ML estimators:
 * They might behave badly if the individual featrues do not more or less look like standard
 * normally distributed data.
 *
 * For instance many elements used in the objective funtion of a learning algorithm assume
 * that all features are centered around 0 and have variance in the same order.
 * If a feature has a variance that is orders of maginiteds larger than others,
 * it might dominate the objective function and make the estimator unable to
 * learn from other feaztures correctly as expected.[^1]
 *
 *
 *
 *
 * - [^1]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
 */
#include "../../Matrix.h"
#include "../../numerics/utils.h"
#include "../Predictor.h"

/**
 *
 *
 *
 */

class StandardScaler : public Transformer
{
private:
  bool with_std   = true;
  bool with_means = true;

public:
  Matrix<double> means;
  Matrix<double> std_deviations;

public:
  StandardScaler(bool withMeans = true, bool withStd = true)
    : Transformer()
    , with_std(withStd)
    , with_means(withMeans){};

  /**
   * Computes mean and sds to be used later in scaling.
   *
   * @param X: given matrix to use to calculate mean/standard deviations
   * @param y: unused
   */
  void fit(const Matrix<double>& X, [[maybe_unused]] const Matrix<double>& y) override {
    means          = with_means ? mean(X, 0) : zeros(1, X.columns());
    std_deviations = with_std ? sd(X, 0) : ones(1, X.columns());
  }

  /**
   * Perform standarization by centering and scaling
   * *
   * To achieve center at 0 and unit variance.
   *
   * @param in: Data to transform
   * @returns: transformed data
   */
  Matrix<double> transform(const Matrix<double>& in) override {
    auto diff = in - means;
    return diff / std_deviations;
  }
};

class MinMaxScaler : public Transformer
{
private:
  double min_val;
  double max_val;

public:
public:
  MinMaxScaler(double range_min_val = 0., double range_max_val = 1.0)
    : Transformer()
    , min_val(range_min_val)
    , max_val(range_max_val) { }

  void fit(const Matrix<double>& X, [[maybe_unused]] const Matrix<double>& y) override { }

  Matrix<double> transform(const Matrix<double>& in) override {
    auto localDiff  = max(in, 0) - min(in, 0);
    auto shiftedIn  = in - min(in, 0);
    auto scaledDiff = shiftedIn / localDiff;
    scaledDiff      = scaledDiff * (max_val - min_val);
    scaledDiff      = scaledDiff + Matrix<double>(min_val, scaledDiff.rows(), scaledDiff.columns());
    return scaledDiff;
  }
};

/**
 * \example ds/preprocessing/TestScaler.cpp
 * This is an example on how to use the Scaler classes
 */
