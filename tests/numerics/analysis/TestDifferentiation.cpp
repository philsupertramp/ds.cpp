#include "../../Test.h"
#include <math/numerics/analysis/Differentiation.h>
#include <math/visualization/Plot.h>


class DifferentiationTestCase : public Test
{
  int m             = 10;
  double a          = 0;
  double b          = 2 * M_PI;
  Matrix<double> xk = linspace(a, b, m).Transpose();

  static Matrix<double> f(const Matrix<double>& in) {
    return in.Apply([](const double& val) { return cos(val); });
  }

  static Matrix<double> df(const Matrix<double>& in) {
    return in.Apply([](const double& val) { return -sin(val); });
  }

  bool TestDiffs() {
    auto expected = df(xk);
    auto f_xk     = f(xk);

    Plot plot("diffs");
    plot.AddData(HorizontalConcat(xk, expected), "Analytical result");
    plot.AddData(HorizontalConcat(xk, forwardDiff(xk, f(xk))), "forward", DataTypes::DOTS, "x");
    plot.AddData(HorizontalConcat(xk, backwardDiff(xk, f(xk))), "backward", DataTypes::DOTS, "o");
    plot.AddData(HorizontalConcat(xk, centralDiff(xk, f(xk))), "centralDiff", DataTypes::DOTS, "+");
    plot.AddData(HorizontalConcat(xk, backwardDiff2(xk, f(xk))), "backward2", DataTypes::DOTS, "-");
    plot.AddData(HorizontalConcat(xk, centralDiff4(xk, f(xk))), "centralDiff4", DataTypes::DOTS, "#");
    plot();

    return true;
  }
  bool TestBackwardDiffs() {
    auto expected = df(xk);
    auto f_xk     = f(xk);
    Matrix<double> backward({
    { 0 },
    { -0.33511665007288566 },
    { -0.84854514524284763 },
    { -0.9649299364248537 },
    { -0.6298132863519682 },
    { 0 },
    { 0.62981328635196787 },
    { 0.9649299364248537 },
    { 0.84854514524284785 },
    { 0.33511665007288594 },
    });
    AssertEqual(backwardDiff(xk, f_xk), backward);
    return true;
  }

  bool TestForwardDiffs() {
    auto expected = df(xk);
    auto f_xk     = f(xk);
    Matrix<double> forward({
    { -0.33511665007288566 },
    { -0.84854514524284763 },
    { -0.9649299364248537 },
    { -0.6298132863519682 },
    { 0 },
    { 0.62981328635196787 },
    { 0.9649299364248537 },
    { 0.84854514524284785 },
    { 0.33511665007288594 },
    { 0 },
    });
    AssertEqual(forwardDiff(xk, f_xk), forward);

    return true;
  }
  bool TestCentralDiffs() {
    auto expected = df(xk);
    auto f_xk     = f(xk);
    Matrix<double> central({
    { 0 },
    { -0.59183089765786667 },
    { -0.90673754083385083 },
    { -0.79737161138841095 },
    { -0.31490664317598421 },
    { 0.31490664317598377 },
    { 0.79737161138841095 },
    { 0.90673754083385083 },
    { 0.59183089765786689 },
    { 0 },
    });
    AssertEqual(centralDiff(xk, f_xk), central);
    return true;
  }
  bool TestBackwardDiffs2() {
    auto expected = df(xk);
    auto f_xk     = f(xk);
    Matrix<double> backward2({
    { 0 },
    { 0 },
    { -1.1052593928278287 },
    { -1.0231223320158569 },
    { -0.46225496131552568 },
    { 0.31490664317598388 },
    { 0.94471992952795147 },
    { 1.132488261461297 },
    { 0.79035274965184488 },
    { 0.078402402487904935 },
    });
    AssertEqual(backwardDiff2(xk, f_xk), backward2);
    return true;
  }

  bool TestCentralDiffs4() {
    auto expected = df(xk);
    auto f_xk     = f(xk);
    Matrix<double> central4({ { 0 },
                              { 0 },
                              { -0.97744963627075465 },
                              { -0.85955478451624223 },
                              { -0.33946469619924124 },
                              { 0.33946469619924052 },
                              { 0.8595547845162419 },
                              { 0.9774496362707551 },
                              { 0 },
                              { 0 } });

    AssertEqual(centralDiff4(xk, f_xk), central4);
    return true;
  }

public:
  virtual void run() {
#if USE_VIS
    TestDiffs();
#endif
    TestBackwardDiffs();
    TestBackwardDiffs2();
    TestForwardDiffs();
    TestCentralDiffs();
    TestCentralDiffs4();
  }
};


int main() {
  DifferentiationTestCase().run();
  return 0;
}
