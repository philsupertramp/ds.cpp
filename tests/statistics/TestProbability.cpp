#include "../Test.h"
#include <math/statistics/Probability.h>

class ProbabilityTestCase : public Test
{
  bool TestLM() {
    Matrix<double> x = { { 1585., 1819., 1647., 1496., 921., 1278., 1810., 1987., 1612., 1413. } };
    Matrix<double> y = { { 12.5, 10., 9.95, 11.5, 12., 10., 8., 9., 9.5, 12.5 } };
    x                = x.Transpose();
    y                = y.Transpose();

    auto out = lm(x, y);
    AssertEqual(out.beta_0, -0.0031336232606394786);
    AssertEqual(out.beta_1, 15.373424692163542);

    return true;
  }

  bool TestRound() {
    AssertEqual(round(0.00001, 4), 0);
    AssertEqual(round(0.000011, 5), 0.00001);
    AssertEqual(round(0.1, 0), 0.0);

    return true;
  }
  bool TestABS() {
    AssertEqual(abs(-1.0), 1.0);
    AssertEqual(abs(0.0), 0.0);
    AssertEqual(abs(1.0), 1.0);

    return true;
  }
  bool TestPow() {
    AssertEqual(pow(2, 2), 4);
    AssertEqual(pow(5, 2), 25);
    AssertEqual(pow(1, 20), 1);
    AssertEqual(pow(-1, 20), 1);
    AssertEqual(pow(-1, 21), -1);
    return true;
  }
  bool TestNorm() {
    AssertEqual(norm({ { 1.0 } }, P_NORM::Inf), 1.0);
    AssertEqual(norm({ { 1.0, 10.0 } }, P_NORM::Inf), 10.0);
    AssertEqual(norm({ { 1.0, 10.0 } }, P_NORM::One), 11.0);
    AssertEqual(norm({ { 1.0, 1.0 } }, P_NORM::Eukl), sqrt(2.0));
    return true;
  }
  bool TestCov() { return true; }
  bool TestVar() { return true; }
  bool TestCoefficientOfDetermination() { return true; }
  bool TestGetExponent() { return true; }
  bool TestLikelihood() { return true; }


  bool TestRegression() {
    Matrix<double> A = { { -1, 0, 3, 7, 4, 6, 10 } };
    auto out         = Regression(A.Transpose());

    Matrix<double> expected = { { -0.7857142857142847,
                                  0.8571428571428577,
                                  2.5,
                                  4.142857142857142,
                                  5.785714285714285,
                                  7.428571428571427,
                                  9.07142857142857 } };

    AssertEqual(out, expected.Transpose());

    return true;
  }

  bool Testsd() {
    Matrix<double> A = { { -1, 0, 3, 7, 4, 6, 10 } };
    Matrix<double> B = {
      { -1, 0, 3, 7, 4, 6, 10 },
      { 12.5, 10., 9.95, 11.5, 12., 10., 8. },
    };

    auto res  = sd(A.Transpose(), 0);
    auto res2 = sd(B.Transpose(), 0);

    AssertEqual(res(0, 0), 3.6027200608339);
    AssertEqual(res2(0, 0), 3.6027200608339);
    AssertEqual(res2(0, 1), 1.4279641566416);


    return true;
  }

  bool TestCorr() {
    Matrix<double> A = { { 1, 2, 3 } };
    Matrix<double> B = { { 1, 2, 3 } };

    std::cout << corr(A.Transpose(), B.Transpose());

    return true;
  }

public:
  virtual void run() {
    TestRound();
    TestABS();
    TestPow();
    TestNorm();
    TestCov();
    TestVar();
    TestLM();
    TestCoefficientOfDetermination();
    TestGetExponent();
    TestLikelihood();
    TestRegression();
    Testsd();
    TestCorr();
  }
};

int main() {
  ProbabilityTestCase().run();
  return 0;
}
