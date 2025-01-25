#include "../Matrix.h"


class Predictor
{
public:
  /**
   * Implements training algorithm
   * @param X: array-like with the shape: [n_samples, n_features]
   * @param y: array-like with shape: [n_samples, 1]
   * @returns this
   */
  virtual void fit(const Matrix<double>& X, const Matrix<double>& y) = 0;

  /**
   * Makes prediction for given input
   * @returns
   */
  virtual Matrix<double> predict(const Matrix<double>&) = 0;

  /**
   *
   *
   */
  virtual Matrix<double> transform(const Matrix<double>&) = 0;
};


class Transformer : public Predictor
{
public:
  Matrix<double> predict(const Matrix<double>& in) override { return in; };
};
