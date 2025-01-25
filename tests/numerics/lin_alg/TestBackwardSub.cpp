#include "../../Test.h"
#include <math/numerics/lin_alg/backwardSub.h>
#include <vector>


class BackwardSubTestCase : public Test
{
  bool TestBackwardSub() {
    Matrix<double> R = { { 15.0, 2.0 }, { 16.0, 46.0 } };
    Matrix<double> b = { { 2.0 }, { 3.0 } };
    auto x           = backwardSub(R, b);

    std::vector<double> LsgX = { 0.1246376811594203, 0.0652173913043478 };
    for(size_t i = 0; i < 2; i++) { AssertEqual(x(i, 0), LsgX[i]); }


    R = { { 4, 2, 3 }, { 0, 1, -0.5 }, { 0, 0, 1 } };
    b = { { 2 }, { 0 }, { 1 } };
    x = backwardSub(R, b);

    LsgX = { -0.5, 0.5, 1.0 };

    for(size_t i = 0; i < 3; i++) { AssertEqual(x(i, 0), LsgX[i]); }

    return true;
  }

  bool TestErrors() {
    Matrix<double> R = { { 15.0 }, { 16.0 } };
    Matrix<double> b = { { 2.0 }, { 3.0 } };

    auto x = backwardSub(R, b);
    assert(x.rows() == 0);

    Matrix<double> R2 = { { 15.0, 1.0 } };
    Matrix<double> b2 = { { 2.0 }, { 3.0 } };

    auto x2 = backwardSub(R2, b2);
    assert(x2.rows() == 0);
    assert(x2.columns() == 0);
    return true;
  }

public:
  void run() override {
    TestBackwardSub();
    TestErrors();
  }
};

int main() { BackwardSubTestCase().run(); }