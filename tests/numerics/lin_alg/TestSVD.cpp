#include "../../Test.h"
#include <math/numerics/lin_alg/svd.h>
#include <math/numerics/utils.h>
#include <math/visualization/Plot.h>


class SVDTestCase : public Test
{
  bool TestSVDDimensions() {
    Matrix<double> A = { { 1, 1, 1, 1 }, { 1, 3, 1, 2 } };
    auto result      = svd(A.Transpose(), 0);
    auto U           = result[0];
    auto S           = eye(U.rows(), U.columns());
    S(0, 0)          = result[1](0, 0);
    S(1, 1)          = result[1](0, 1);
    auto VH          = result[2];
    // A stays A
    AssertEqual((U.Transpose() * S) * VH, A);
    return true;
  }
  bool TestSVD() {
    Matrix<double> A = { { 1, 1, 1, 1 }, { 1, 3, 1, 2 } };
    auto result      = svd(A, 0);
    auto U           = result[0];
    auto S           = eye(U.rows(), U.columns());
    S(0, 0)          = result[1](0, 0);
    S(1, 1)          = result[1](0, 1);
    auto VH          = result[2];
    // A stays A
    AssertEqual(U * S * VH, A);

    return true;
  }

  /**
   * For more details see https://wiki.godesteem.de/wiki/pca-using-svd/
   * @returns
   */
  bool TestPCANormalDistribution() {
    // Create in (2, 1) centered normal distributed data.
    size_t n_points    = 1000;
    auto random_data   = Matrix<double>::Normal(n_points, 2, 0, 0.1).Transpose();
    Matrix<double> xC  = { { 2 }, { 1 } };   // center (mean)
    Matrix<double> sig = { { 2 }, { 0.5 } }; // principal axes

    // Rotate data
    auto theta = M_PI / 3.0;

    // Transformation matrix for rotation
    Matrix<double> R = { { cos(theta), -sin(theta) }, { sin(theta), cos(theta) } };

    // Transform normal distributed data
    auto X = (R * diag(sig) * random_data + diag(xC) * ones(2, n_points));

    // 1. compute mean row
    Matrix<double> row_mean(0.0, 1, X.rows());
    for(size_t i = 0; i < X.rows(); ++i) { row_mean(0, i) = mean(X.GetSlice(i, i, 0, X.columns() - 1), -1)(0, 0); }
    Matrix<double> X_bar = (Matrix<double>(1.0, X.columns(), 1) * row_mean).Transpose();

    // 2. B = X - X'
    auto B = X - X_bar;
    B      = 1.0 / (double)sqrt(static_cast<double>(X.rows())) * B;

    // find principal components
    auto SVD = svd(B, 0);

    auto U  = SVD[0].Transpose();
    auto S  = SVD[1];
    auto VT = SVD[2];

    Matrix<double> Theta = 2.0 * M_PI * linspace(0, 1, 100).Transpose();
    auto newTheta =
    HorizontalConcat(Theta.Apply([](double val) { return cos(val); }), Theta.Apply([](double val) { return sin(val); }))
    .Transpose();
    auto xStd = U * diag(S) * newTheta;
#if USE_VIS
    Plot plot1("PCA Example");
    plot1.AddData(random_data.Transpose(), "Data", DOTS);
    plot1();
    plot1.AddData(X.Transpose(), "Data", DOTS);
    plot1();

    Plot plot("PCA Example");
    plot.AddData(X.Transpose(), "Data", DOTS);

    auto mean1 = row_mean(0, 0);
    auto mean2 = row_mean(0, 1);

    // 1-a confidence interval
    plot.AddData(
    xStd.GetSlice(0, 0, 0, xStd.columns() - 1).Apply([mean1](double v) { return mean1 + v; }).Transpose(),
    xStd.GetSlice(1, 1, 0, xStd.columns() - 1).Apply([mean2](double v) { return mean2 + v; }).Transpose(),
    "conf 1",
    LINE);
    plot.AddData(
    xStd.GetSlice(0, 0, 0, xStd.columns() - 1).Apply([mean1](double v) { return mean1 + 2.0 * v; }).Transpose(),
    xStd.GetSlice(1, 1, 0, xStd.columns() - 1).Apply([mean2](double v) { return mean2 + 2.0 * v; }).Transpose(),
    "conf 2",
    LINE);
    plot.AddData(
    xStd.GetSlice(0, 0, 0, xStd.columns() - 1).Apply([mean1](double v) { return mean1 + 3.0 * v; }).Transpose(),
    xStd.GetSlice(1, 1, 0, xStd.columns() - 1).Apply([mean2](double v) { return mean2 + 3.0 * v; }).Transpose(),
    "conf 3",
    LINE);

    std::cout << "S:\n" << S << std::endl;

    Matrix<double> pc1 = {
      { mean1, mean2 },
      { mean1 + U(0, 0) * S(0, 0), mean2 + U(1, 0) * S(0, 0) },
    };
    std::cout << "PC1:\n" << pc1 << std::endl;
    plot.AddData(pc1, "pc 1", LINE);
    Matrix<double> pc2 = { { mean1, mean2 }, { mean1 + U(0, 1) * S(0, 1), mean2 + U(1, 1) * S(0, 1) } };
    std::cout << "PC2:\n" << pc2 << std::endl;
    plot.AddData(pc2, "pc 2", LINE);
    plot();
#endif
    return true;
  }

public:
  void run() override {
    TestSVD();
    //TestSVDDimensions();
    TestPCANormalDistribution();
  }
};

int main() {
  SVDTestCase().run();
  return 0;
}
