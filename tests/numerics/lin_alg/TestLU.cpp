#include "../../Test.h"
#include <math/numerics/lin_alg/LU.h>
#include <math/numerics/utils.h>
#include <vector>


class LUTestCase : public Test
{
  bool TestLU() {
    Matrix<double> A = { { 4, 2, 3 }, { 2, 2, 1 }, { 2, 2, 2 } };
    Matrix<double> B = { { 4, 2, 3 }, { 2, 2, 1 }, { 2, 2, 2 } };

    auto result = LU(A);

    // A stays A
    AssertEqual(A, B);

    std::vector<unsigned int> p = { 0, 1, 2 };
    Matrix<double> LR           = { { 4.0, 2.0, 3.0 }, { 0.5, 1.0, -0.5 }, { 0.5, 1.0, 1.0 } };

    for(size_t col = 0; col < 3; col++) {
      assert(result.second[col] == p[col]);
      for(size_t row = 0; row < 3; row++) { AssertEqual(result.first(col, row), LR(col, row)); }
    }
    return true;
  }

  bool TestLUPivot() {
    Matrix<double> A = { { 2, 2, 1 }, { 4, 2, 3 }, { 2, 2, 2 } };

    auto result = LU(A);

    std::vector<unsigned int> p = { 1, 0, 2 };
    Matrix<double> LR           = { { 4.0, 2.0, 3.0 }, { 0.5, 1.0, -0.5 }, { 0.5, 1.0, 1.0 } };

    for(size_t col = 0; col < 3; col++) {
      assert(result.second[col] == p[col]);
      for(size_t row = 0; row < 3; row++) { AssertEqual(result.first(col, row), LR(col, row)); }
    }
    return true;
  }

public:
  void run() override {
    TestLU();
    TestLUPivot();
  }
};

int main() {
  LUTestCase().run();
  return 0;
}