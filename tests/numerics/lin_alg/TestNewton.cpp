#include "../../Test.h"
#include <cmath>
#include <math/numerics/lin_alg/newton.h>
#include <vector>


class NewtonTestCase : public Test
{
  bool TestNewton() {
    auto f  = [](Matrix<double> x) { return Matrix<double>({ { cos(x(0, 0)) - x(0, 0) } }); };
    auto Df = [](Matrix<double> x) { return Matrix<double>({ { (-sin(x(0, 0)) - 1) } }); };

    int maxIter = 100;
    double TOL  = 1e-16;
    Matrix<double> x0({ { 1 } });
    auto res = newton(f, Df, x0, TOL, maxIter);

    std::vector<double> xExpt = { 0.73908513321516067 };

    for(int i = 0; i < 1; ++i) { AssertEqual(res.first(i, 0), xExpt[i]); }
    return true;
  }

public:
  void run() override { TestNewton(); }
};

int main() { NewtonTestCase().run(); }
