#include "../Predictor.h"


class OneHotEncoder : public Transformer
{
  Matrix<int> unique_values;

public:
  OneHotEncoder()
    : Transformer() { }

  void fit(const Matrix<double>& X, [[maybe_unused]] const Matrix<double>& y) override { unique_values = unique(X, 0); }

  Matrix<double> transform(const Matrix<double>& in) override { return in; }
};
