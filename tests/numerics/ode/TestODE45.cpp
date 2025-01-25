#include "../../Test.h"
#include <math/numerics/ode/ODESolver.h>

class ODE45TestCase : public Test
{
  bool TestOde45() {
    auto ode                      = []([[maybe_unused]] double t, const Matrix<double>& y) { return y; };
    std::vector<double> tInterval = { 0.0, 2.0 };
    Matrix<double> y0             = { { 1.0 } };
    double h                      = 0.1;

    auto foo = ODESolver::ode45(ode, tInterval, y0, { h });

    auto yResult = foo.Y;
    auto tResult = foo.T;

    //        std::cout << yResult;

    double tExpected[21] = { 0.0,
                             0.100000000000000,
                             0.200000000000000,
                             0.300000000000000,
                             0.400000000000000,
                             0.500000000000000,
                             0.600000000000000,
                             0.700000000000000,
                             0.800000000000000,
                             0.900000000000000,
                             1.000000000000000,
                             1.100000000000000,
                             1.200000000000000,
                             1.300000000000000,
                             1.400000000000000,
                             1.500000000000000,
                             1.600000000000000,
                             1.700000000000000,
                             1.800000000000000,
                             1.900000000000000,
                             2.000000000000000 };
    double yExpected[21] = {
      1.000000000000000, 1.105170918333333, 1.221402758729743, 1.349858808520217, 1.491824699032628, 1.648721272622238,
      1.822118802939620, 2.013752710757214, 2.225540932643790, 2.459603116318360, 2.718281834797091, 3.004166031651519,
      3.320116932026115, 3.669296678741350, 4.055199980082028, 4.481689086012570, 4.953032442872989, 5.473947413424735,
      6.049647489802989, 6.685894471898514, 7.389056133387838,
    };

    for(int i = 0; i < 21; i++) {
      AssertEqual(tExpected[i], tResult(i, 0));
      AssertEqual(yExpected[i], yResult(i, 0));
    }
    return true;
  }


  bool TestOde45RB() {
    auto ode = []([[maybe_unused]] double t, const Matrix<double>& y) {
      Matrix<double> result(0, y.rows(), y.columns());

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
    auto foo                      = ODE45(ode, tInterval, y0, h);

    auto yResult = foo.Y;
    auto tResult = foo.T;

    double yExpected[6][2] = {
      { 80.0000, 30.0000 },
      { 78.360998725243107, 24.268754512660799 },
      { 80.847619520187592, 19.730492689186242 },
      { 86.622395662576622, 16.728217389452837 },
      { 94.917043272614094, 15.226418795854334 },
      { 104.81087505230401, 15.192897445498955 },
    };
    double tExpected[6] = { 0, 1, 2, 3, 4, 5 };

    for(int i = 0; i < 6; i++) {
      AssertEqual(tExpected[i], tResult(i, 0));
      AssertEqual(yExpected[i][0], yResult(i, 0));
      AssertEqual(yExpected[i][1], yResult(i, 1));
    }
    return true;
  }

public:
  void run() override {
    TestOde45();
    TestOde45RB();
  }
};

int main() {
  ODE45TestCase().run();
  return 0;
}
