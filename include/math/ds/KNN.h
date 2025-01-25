/**
 * K Nearest Neightbor Classifier (KNN) for linear and non-linear classification.
 *
 * Works well on
 * - correlated data
 * - uncorrelated data
 *
 *
 * Implementation:
 *
 * fit model:
 *  not required
 *
 * prediction:
 *  1. Compute distance between test data X and training data weights: $||weights - X_i||_2$
 *  2. Sort distances, in ascending order, select K shortest distances: $\text{argsort} ||weights - X_i||_2$
 *  3. For each class in trainLabels count the number of members inside the distance set: $\gamma_{y_k} \leftarrow \gamma_{y_k} + 1$
 *  4. Select the class with most members inside distance set: $\text{argmax}_k(\gamma)

 *
 */

#include "../Matrix.h"
#include "../matrix_utils.h"
#include "../numerics/utils.h"
#include "Classifier.h"
#include <cstddef>


class KNN : public Classifier
{
  int nearest_neighbors;
  Matrix<double> trainLabels;

public:
  KNN(int neighbors)
    : Classifier()
    , nearest_neighbors(neighbors) { }


  /**
   * Helper method to 'fit' the model.
   *
   * Stores given X in weights and y in trainLabels
   *
   * @param X Matrix of known input data
   * @param y vector of corresponding labels
   */
  void fit(const Matrix<double>& X, const Matrix<double>& y) override {
    assert(y.rows() == 1 || y.columns() == 1);
    weights     = X;
    trainLabels = y;
  };

  /**
   *  K-Nearest Neighbor Classifier Prediction
   *
   *  1. Compute distance between test data X and training data weights: $||weights - X_i||_2$
   *  2. Sort distances, in ascending order, select K shortest distances: $\text{argsort} ||weights - X_i||_2$
   *  3. For each class in trainLabels count the number of members inside the distance set: $\gamma_{y_k} \leftarrow
   * \gamma_{y_k} + 1$
   *  4. Select the class with most members inside distance set: $\text{argmax}_k(\gamma)
   *
   */
  Matrix<double> predict(const Matrix<double>& x) override {
    auto predictions = zerosV(x.rows());
    auto trainX      = weights;
    auto trainY      = trainLabels;
    auto labels      = unique(trainY);
    for(size_t i = 0; i < x.rows(); ++i) {
      auto xi            = x.GetSlice(i);
      auto current_dists = norm(trainX - xi, 0);
      auto k_neares      = trainY.GetSlicesByIndex(argsort(current_dists));
      auto k_nearest     = k_neares.GetSlice(0, nearest_neighbors - 1);
      auto gamma         = zeros(labels.rows(), 1);
      for(size_t k = 0; k < labels.rows(); ++k) {
        std::function<bool(double)> condition = [k](double x) { return bool(x == k); };
        // O(N)
        auto yis = where(condition, k_nearest, { { 1 } }, { { 0 } });
        // O(N)
        gamma(k, 0) = yis.sumElements();
      }
      predictions(i, 0) = argmax(gamma);
    }
    return predictions;
  };
};
/**
 * \example ds/TestKNN.cpp
 * This is an example on how to use the KNN class for classification.
 */
