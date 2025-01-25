#include "../../Test.h"
#include <math/numerics/lin_alg/qr.h>
#include <math/numerics/utils.h>


class QRTestCase : public Test
{
  bool TestQR() {
    Matrix<double> A = { { 2, 2, 8 }, { 5, 6, 6 }, { 4, 4, 8 } };
    //
    //        R *= -1.0;
    //        Q *= -1.0;
    auto result = qr(A);
    AssertEqual(A, result.first * result.second);

    A = { { 2, 3 }, { 3, 2 } };

    result = qr(A);

    AssertEqual(A, result.first * result.second);

    return true;
  }

public:
  void run() override { TestQR(); }
};

int main() {
  QRTestCase().run();
  return 0;
}