#include "../Test.h"
#include <math/statistics/Insurance.h>

class InsuranceTestCase : public Test
{
  bool TestCompoundingFactor() {
    AssertEqual(compoundingFactor(0.01), 1.01);
    AssertEqual(compoundingFactor(1.01), 2.01);
    return true;
  }
  bool TestDiscountFactor() {
    AssertEqual(discountFactor(1.0), 0.5);
    AssertEqual(discountFactor(0.1), 1. / 1.1);
    return true;
  }
  bool TestFundamentalValue() {
    Interest I = { 0.06, 100, 1, 10, 5 };
    // 10 * 1.06^(-100)
    AssertEqual(fundamentalValue(I), 0.029472262287399215);
    return true;
  }

  bool TestEndValue() {
    Interest I = { 0.06, 100, 1, 10, 5 };
    // 1 * 1.06^(100)
    AssertEqual(endValue(I), 339.3020835144872);
    return true;
  }
  bool TestTermInPeriods() {
    // (log(10)-log(1))/log(1.05)
    Interest I = { 0.06, 100, 1, 10, 5 };
    AssertEqual(termInPeriods(I), 39.516530635771536208682805631274488204856495225826522986578631731);
    return true;
  }
  bool TestFundamentalValueOfAnnuityInAdvance() {
    Interest I = { 0.06, 100, 1, 10, 5 };
    // (1-pow(1/1.06, 100))/(1-1/1.06)
    AssertEqual(
    fundamentalValueOfAnnuityInAdvance(I), 17.614599003292261124646315796916331824982312779861072540295575993);
    return true;
  }
  bool TestEndValueOfAnnuityInAdvance() {
    Interest I = { 0.06, 100, 1, 10, 5 };
    // (pow(1.06, 100)-1)/(1-1/1.06)
    AssertEqual(endValueOfAnnuityInAdvance(I), 5976.6701420892436797668612701763787396284937497058441484295690490);
    return true;
  }
  bool TestFundamentalValueOfAnnuityToArrear() {
    Interest I = { 0.06, 100, 1, 10, 5 };
    // (1-pow(1/1.06, 100))/(0.06)
    AssertEqual(
    fundamentalValueOfAnnuityInArrear(I),
    16.617546229521001060987090374449369646209729037604785415373184899351288068383264304267210888801306835712115089689784848248668120);
    return true;
  }
  bool TestEndValueOfAnnuityInArrear() {
    Interest I = { 0.06, 100, 1, 10, 5 };
    // (1-pow(1/1.06, 100))/(0.06) * pow(1.06, 100)
    AssertEqual(endValueOfAnnuityInArrear(I), 5638.3680585747581884593030850720554147438620280243812721033670274);
    return true;
  }

public:
  virtual void run() {
    TestCompoundingFactor();
    TestDiscountFactor();
    TestFundamentalValue();
    TestEndValue();
    TestTermInPeriods();
    TestFundamentalValueOfAnnuityInAdvance();
    TestEndValueOfAnnuityInAdvance();
    TestFundamentalValueOfAnnuityToArrear();
    TestEndValueOfAnnuityInArrear();
  }
};

int main() {
  InsuranceTestCase().run();
  return 0;
}
