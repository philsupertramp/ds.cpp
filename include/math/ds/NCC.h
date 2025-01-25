/**
 * Nearest Centroid Classifier:
 *
 * Linear classifier for linear seperable/uncorrelated data.
 *
 * Prediction:
 *   In:
 *     $x$: Data point to classify
 *     $\mu_k$: $k$ class centroids
 *   Out: $$
 *     argmin(||\mu_k - x||_2)
 *   $$
 *
 * Compute centroids/fit model:
 *
 *   Batch-Mode:
 *     In:
 *       $x_i$ with $x_i\in \R^D$: $N$ training samples
 *       $y_i$ with $y_i\in \{1, ..., K\}$: $N$ training sample lables
 *     Out:
 *       $\mu_k$: $k$ class centroids
 *
 *     Computes **all** centroids at once, holds $N$ $D$-dimensional samples in memory at-a-time.
 *
 *   Iterative/Streaming-Mode:
 *     In:
 *       $x_i$ with $x_i\in \R^D$: $N$ training samples
 *       $y_i$ with $y_i\in \{1, ..., K\}$: $N$ training sample lables
 *     Out:
 *       $\mu_k$: $K$ class centroids
 *
 *     Calculates centroids iteratively, holds $K$ centroids and 1 training sample in memory at-a-time.
 *
 *
 */

#include "../matrix_utils.h"
#include "../numerics/utils.h"
#include "Classifier.h"
#include <cstddef>


/**
 *
 *
 */
class NCC : public Classifier
{
  bool use_iterative = true;

public:
  NCC(bool useIterative)
    : Classifier()
    , use_iterative(useIterative) { }

  /**
   *
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    assert(y.rows() == 1 || y.columns() == 1);
    auto labels = unique(y);
    weights     = zeros(labels.rows(), X.columns());
    if(use_iterative) {
      fit_iterative(X, y, labels);
    } else {
      fit_batch(X, y, labels);
    }
  };

  /**
   * Batch-wise implementation of nearest centroid classifier centroid calculation.
   *
   * Note: Requires all input data in memory.
   *
   * Complexity: O(K * 6N)
   *
   * @param X given input data to calculate centroids from
   * @param y given labels for input data X
   * @param labels unique labels inside y
   */
  void fit_batch(const Matrix<double>& X, const Matrix<double>& y, const Matrix<double>& labels) {
    // TODO: Technically we can omit labels here and only pass the number of unique labels. Or we keep a vector
    // of unique labels as a class attribute. Both options might make sense.
    // O(K)
    for(size_t i = 0; i < labels.rows(); ++i) {
      std::function<bool(double)> condition = [i](double x) { return bool(x == i); };
      // O(N)
      auto yis = where(condition, y, { { 1 } }, { { 0 } });
      // O(N)
      auto Nk = yis.sumElements();
      // O(2N)
      auto Xis = X.GetSlicesByIndex(where_true(yis));
      // O(N * N * N)
      weights.SetSlice(i, Xis.sum(1) * (1. / Nk));
    }
  }

  /**
   *
   * Iterative implementation of nearest centroid classifier centroid calculation
   *
   * Note: Allows streaming of input data.
   *
   * Complexity: O(N * 3D)
   *
   * @param X input data to calculate centroids from
   * @param y given labels for input data X
   * @param labels unique labels inside y
   */
  void fit_iterative(const Matrix<double>& X, const Matrix<double>& y, const Matrix<double>& labels) {
    auto counters = Matrix<int>(0, labels.rows(), 1);
    // O(N)
    for(size_t i = 0; i < X.rows(); ++i) {
      auto k               = y(i, 0);
      auto xi              = X.GetSlice(i, i);
      auto current_counter = counters(k, 0);
      // O(D)
      weights.SetSlice(
      k,
      // O(D)
      weights.GetSlice(k, k) * (current_counter / double(current_counter + 1))
      // O(D)
      + xi * (1. / double(current_counter + 1)));
      counters(k, 0) += 1;
    }
  }

  /**
    Prediction:
*   In:
*     $x$: Data point to classify
*     $\mu_k$: $k$ class centroids
*   Out: $$
*     argmin(||\mu_k - x||_2)
*   $$
*
    *
    *
    */
  Matrix<double> predict(const Matrix<double>& x) override {
    auto predictions = zerosV(x.rows());
    for(size_t i = 0; i < x.rows(); ++i) {
      auto xi           = x.GetSlice(i, i);
      auto distances    = norm(weights - xi, 0);
      predictions(i, 0) = argmin(distances);
    }
    return predictions;
  };
};
/**
 * \example ds/TestNCC.cpp
 * This is an example on how to use the NCC class for classification.
 */
