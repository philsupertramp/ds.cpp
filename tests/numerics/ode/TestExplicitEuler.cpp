#include "../../Test.h"
#include <math/numerics/ode/ODESolver.h>


class ODEExplicitEulerTestCase : public Test
{
  bool TestExplicitEuler() {
    auto ode                      = []([[maybe_unused]] double t, const Matrix<double>& y) { return y; };
    std::vector<double> tInterval = { 0.0, 5.0 };
    Matrix<double> y0             = { { 5.0 } };
    double h                      = 1.0;
    ODEOption option              = { h };
    ODEResult foo                 = ODESolver::odeExpEuler(ode, tInterval, y0, option);

    auto yResult = foo.Y;
    auto tResult = foo.T;

    double tExpected[6] = { 0, 1, 2, 3, 4, 5 };
    double yExpected[6] = { 5, 10, 20, 40, 80, 160 };
    for(int i = 0; i < 6; i++) {
      AssertEqual(tExpected[i], tResult(i, 0));
      AssertEqual(yExpected[i], yResult(i, 0));
    }
    return true;
  }

  bool TestExplicitEulerRB() {
    auto ode = []([[maybe_unused]] double t, const Matrix<double>& y) {
      Matrix<double> result(0, y.rows(), y.columns(), 1);

      auto alpha = 0.25;
      auto beta  = -0.01;
      auto gamma = -1.0;
      auto delta = 0.01;

      result(0, 0) = alpha * y(0, 0) + beta * y(0, 0) * y(0, 1);
      result(0, 1) = gamma * y(0, 1) + delta * y(0, 0) * y(0, 1);
      return result;
    };
    std::vector<double> tInterval = { 0.0, 5.0 };
    Matrix<double> y0             = { { 80, 30 } };
    double h                      = 1.0;
    auto foo                      = ODEExpEuler(ode, tInterval, y0, h);

    auto yResult = foo.Y;
    auto tResult = foo.T;

    double yExpected[6][2] = { { 80.0000, 30.0000 },
                               { 76.0000, 24.0000 },
                               { 76.759999999999991, 18.240000000000002 },
                               { 81.948975999999988, 14.001024000000001 },
                               { 90.962524202485753, 11.473695797514239 },
                               { 103.26639193637371, 10.436763316733479 } };
    double tExpected[6]    = { 0, 1, 2, 3, 4, 5 };

    for(int i = 0; i < 6; i++) {
      AssertEqual(tExpected[i], tResult(i, 0));
      AssertEqual(yExpected[i][0], yResult(i, 0));
      AssertEqual(yExpected[i][1], yResult(i, 1));
    }
    return true;
  }

public:
  void run() override {
    TestExplicitEuler();
    TestExplicitEulerRB();
  }
};

int main() {
  ODEExplicitEulerTestCase().run();
  return 0;
}