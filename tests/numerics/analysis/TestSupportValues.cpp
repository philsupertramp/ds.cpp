#include "../../Test.h"

#include <math/numerics/analysis/SupportValues.h>

class MonomBaseTestCase : public Test
{
  bool TestConstructor() {
    Matrix<double> X({ { -1 }, { 0 }, { 1 } });
    Matrix<double> Y({ { 1 }, { 0 }, { 1 } });
    MonomBase base(X, Y);

    Matrix<double> a({ { 0 }, { 0 }, { 1 } });
    AssertEqual(base.A, a);

    // Error, leave empty
    X = { { -1, 0, 1, -1 } };
    Y = { { 1 }, { 0 }, { 1 }, { 1 } };
    MonomBase base2(X, Y);

    AssertEqual(base2.A.rows(), 0);
    AssertEqual(base2.A.columns(), 0);

    return true;
  }

  bool TestEvaluate() {
    Matrix<double> X({ { -1 }, { 0 }, { 1 } });
    Matrix<double> Y({ { 1 }, { 0 }, { 1 } });
    MonomBase base(X, Y);

    Matrix<double> xi({ { 0 }, { 2 }, { 6 } });
    Matrix<double> expected = xi.Apply([](const double& in) { return in * in; });

    auto res = base.Evaluate(xi);

    AssertEqual(res, expected);

    return true;
  }

  bool TestFunction() {
    Matrix<double> X({ { -1 }, { 0 }, { 1 } });
    Matrix<double> Y({ { 1 }, { 0 }, { 1 } });
    MonomBase base(X, Y);

    AssertEqual(base.Function(), "0 + 0x^1 + 1x^2");
    return true;
  }

public:
  virtual void run() {
    TestConstructor();
    TestFunction();
    TestEvaluate();
  }
};

class LagrangeBaseTestCase : public Test
{
  bool TestConstructor() {
    Matrix<double> X({ { -1 }, { 0 }, { 1 } });
    Matrix<double> Y({ { 1 }, { 0 }, { 1 } });
    LagrangeBase base(X, Y);

    return true;
  }

  bool TestEvaluate() {
    int m            = 5;
    Matrix<double> X = linspace(-5, 5, m).Transpose();
    Matrix<double> Y = X.Apply([](const double& in) { return in * in; });
    LagrangeBase base(X, Y);

    Matrix<double> xi       = linspace(-3.5, 3.5, 5).Transpose();
    Matrix<double> expected = xi.Apply([](const double& in) { return in * in; });

    auto res = base.Evaluate(xi);

    AssertEqual(res, expected);

    return true;
  }

  bool TestFunction() {
    Matrix<double> X({ { -1 }, { 0 }, { 1 } });
    Matrix<double> Y({ { 1 }, { 0 }, { 1 } });
    LagrangeBase base(X, Y);

    AssertEqual(base.Function(), "1 * (-x/1.000 * -(x - 1.000)/2.000) + 1 * ((x + 1.000)/2.000 * x/1.000)");
    return true;
  }

public:
  virtual void run() {
    TestConstructor();
    TestFunction();
    TestEvaluate();
  }
};
class NewtonBaseTestCase : public Test
{
  bool TestConstructor() {
    Matrix<double> X({ { 0 }, { 1 }, { 2 } });
    Matrix<double> Y({ { -1 }, { 1 }, { 4 } });
    NewtonBase base(X, Y);

    Matrix<double> expected({ { -1 }, { 2 }, { 0.5 } });

    AssertEqual(base.b, expected);
    return true;
  }

  bool TestEvaluate() {
    Matrix<double> X({ { -5 }, { 0 }, { 5 } });
    Matrix<double> Y({ { 25 }, { 0 }, { 25 } });
    NewtonBase base(X, Y);

    Matrix<double> b({ { 25 }, { -5 }, { 1 } });

    AssertEqual(base.b, b);

    Matrix<double> xi = linspace(-2.5, 2.5, 3).Transpose();
    Matrix<double> expected({ { 6.25 }, { 0 }, { 6.25 } });

    auto res = base.Evaluate(xi);

    AssertEqual(res, expected);

    return true;
  }

  bool TestFunction() {
    Matrix<double> X({ { -1 }, { 0 }, { 1 } });
    Matrix<double> Y({ { 1 }, { 0 }, { 1 } });
    NewtonBase base(X, Y);

    AssertEqual(base.Function(), "1.000 + -1.000* (x + 1.000) + 1.000* (x + 1.000)* x");
    return true;
  }

public:
  virtual void run() {
    TestConstructor();
    TestFunction();
    TestEvaluate();
  }
};

int main() {
  MonomBaseTestCase().run();
  LagrangeBaseTestCase().run();
  NewtonBaseTestCase().run();
  return 0;
}