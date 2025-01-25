#include "../../Test.h"
#include <math/numerics/lin_alg/forwardSub.h>
#include <vector>


class ForwardSubTestCase : public Test
{
  bool TestForwardSub() {
    Matrix<double> L({ { 1.0f, 0.0f, 0.0f }, { 0.5f, 1.f, 0.f }, { 0.5f, 1.f, 1.f } });
    Matrix<double> b({ { 2.0f }, { 1.0f }, { 2.0f } });
    auto x = forwardSub(L, b);

    std::vector<double> LsgX = { 2.0f, 0.0f, 1.0f };

    for(size_t i = 0; i < 3; i++) { AssertEqual(x(i, 0), LsgX[i]); }


    L = { { 1, 0, 0, 0 }, { 0.5, 1, 0, 0 }, { 0.5, 0.5, 1, 0 }, { 0.25, 0.5, 0.25, 1.0 } };
    b = { { 2 }, { 2 }, { 2.5 }, { -1.625 } };
    x = forwardSub(L, b);

    LsgX = { 2.0f, 1.0f, 1.0f, -2.8750f };

    for(size_t i = 0; i < 3; i++) { AssertEqual(x(i, 0), LsgX[i]); }

    return true;
  }
  bool TestForwardSubExample() {
    Matrix<double> L = { { 1.0f, 0.0f, 0.0f }, { -2.0f, 3.f, 0.f }, { -1.f, 7.f, 1.f } };
    Matrix<double> b = { { 2.0f }, { -1.0f }, { 4.0f } };
    auto x           = forwardSub(L, b);

    std::vector<double> LsgX = { 2.0f, 3.0f, -15.0f };

    for(size_t i = 0; i < 3; i++) { AssertEqual(x(i, 0), LsgX[i]); }

    return true;
  }

  bool TestErrors() {
    Matrix<double> R({ { 15.0f }, { 16.0f } });
    Matrix<double> b({ { 2.0f }, { 3.0f } });

    auto x = forwardSub(R, b);
    assert(x.columns() == 0);

    Matrix<double> R2 = { { 15.0f, 1.0f } };
    Matrix<double> b2 = { { 2.0f }, { 3.0f } };

    auto x2 = forwardSub(R2, b2);
    assert(x2.rows() == 0);
    assert(x2.columns() == 0);
    return true;
  }

public:
  void run() override {
    TestForwardSub();
    TestForwardSubExample();
    TestErrors();
  }
};

int main() { ForwardSubTestCase().run(); }