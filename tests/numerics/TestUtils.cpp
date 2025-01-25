#include "../Test.h"
#include <math/numerics/utils.h>


class NumUtilsTestCase : public Test
{
  bool TestLinspace() {
    auto vec = linspace(0, 1, 3);
    assert(vec(0, 0) == 0);
    assert(vec(0, 1) == 0.5);
    assert(vec(0, 2) == 1.0);

    vec = linspace(1, 2, 3);
    assert(vec(0, 0) == 1.0);
    assert(vec(0, 1) == 1.5);
    assert(vec(0, 2) == 2.0);
    return true;
  }

  bool TestZeros() {
    auto A = zeros(2, 3);

    AssertEqual(A.rows(), 2);
    AssertEqual(A.columns(), 3);

    for(size_t i = 0; i < A.rows(); i++) {
      for(size_t j = 0; j < A.columns(); ++j) { AssertEqual(A(i, j), 0.0); }
    }
    return true;
  }
  bool TestOnes() {
    auto A = ones(2, 3);
    for(size_t i = 0; i < A.rows(); i++) {
      for(size_t j = 0; j < A.columns(); ++j) { AssertEqual(A(i, j), 1.0); }
    }
    return true;
  }
  bool TestZerosV() {
    auto A = zerosV(3);

    AssertEqual(A.rows(), 3);
    AssertEqual(A.columns(), 1);

    for(size_t i = 0; i < A.rows(); i++) { AssertEqual(A(i, 0), 0.0); }
    return true;
  }
  bool TestEye() {
    auto testEye = [this](const Matrix<double>& B) {
      for(size_t i = 0; i < B.rows(); ++i) {
        for(size_t j = 0; j < B.columns(); ++j) {
          if(i == j) this->AssertEqual(B(i, j), 1.0);
          else
            this->AssertEqual(B(i, j), 0.0);
        }
      }
    };

    auto A = eye(3, 2);

    AssertEqual(A.rows(), 3);
    AssertEqual(A.columns(), 2);
    testEye(A);

    A = eye(2, 3);

    AssertEqual(A.rows(), 2);
    AssertEqual(A.columns(), 3);
    testEye(A);

    A = eye(3);
    AssertEqual(A.rows(), 3);
    AssertEqual(A.columns(), 3);
    testEye(A);

    return true;
  }
  bool TestNorm() {
    Matrix<double> A(0, 2, 2);
    AssertEqual(norm(A), 0.0);

    A = { { 1, 1, 1, 1 } };
    AssertEqual(norm(A), 2.0);

    A = { { 1e-16, 1e-16, 1e-16 } };
    AssertEqual(norm(A), 0.0);
    return true;
  }

public:
  virtual void run() {
    TestEye();
    TestOnes();
    TestZeros();
    TestZerosV();
    TestNorm();
    TestLinspace();
  }
};

int main() {
  NumUtilsTestCase().run();
  return 0;
}
