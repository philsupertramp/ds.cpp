#include "../Test.h"
#include <math/ds/LogRegSGD.h>


class LogRegSGDTestCase : public Test
{
  bool TestLogReg() {
    auto logRegSgd = LogRegSGD(0.15, 25);
    //        Matrix<double> A = Matrix<double>::Random(10, 1, 1, -1, 1);
    Matrix<double> A({
    { 0.29651885541310041 },
    { 0.64581698997155579 },
    { 0.73420390202980546 },
    { -0.33842438014317311 },
    { 0.73079510849631024 },
    { -0.90637656297433933 },
    { -0.58136213318226426 },
    { -0.12520400256483499 },
    { 0.41344311936636324 },
    { 0.0089382791988343868 },
    });
    //        std::cout << A;
    std::function<bool(double)> condition = [](double x) { return bool(x >= 0.0); };
    Matrix<double> B                      = where(condition, A, { { 1 } }, { { -1 } });
    auto maxVal                           = max(A);
    auto minVal                           = min(A);
    A = A.Apply([minVal, maxVal](const double& in) { return (in - minVal) / (maxVal - minVal); });

    logRegSgd.fit(A, B);

    auto val  = logRegSgd.predict({ { -1.0 } })(0, 0);
    auto val2 = logRegSgd.predict({ { 2.0 } })(0, 0);

    AssertEqual(val, 1);   // == 1
    AssertEqual(val2, -1); // == -1
    return true;
  }

  bool TestNoCost() {
    auto logRegSgd = LogRegSGD(0.15, 25);
    AssertEqual(logRegSgd.costFunction({ { 1 } }), 0.0);
    return true;
  }

public:
  void run() override {
    TestLogReg();
    //        TestNoCost();
  }
};

int main() {
  LogRegSGDTestCase().run();
  return 0;
}