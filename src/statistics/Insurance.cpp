#include "../../include/math/statistics/Insurance.h"

double compoundingFactor(double i) { return 1 + i; }
double discountFactor(double i) { return 1. / (1. + i); }
double fundamentalValue(Interest Z) { return Z.B_N * pow(1 + Z.i, -Z.n); }
double endValue(Interest Z) { return Z.B_0 * pow(1 + Z.i, Z.n); }
double termInPeriods(Interest Z) { return (log(Z.B_N) - log(Z.B_0)) / log(compoundingFactor(Z.i)); }
double fundamentalValueOfAnnuityInAdvance(Interest Z) {
  double discount = discountFactor(Z.i);
  return (1 - pow(discount, Z.n)) / (1 - discount);
}
double endValueOfAnnuityInAdvance(Interest Z) {
  double compFactor = compoundingFactor(Z.i);
  double discount   = discountFactor(Z.i);
  return (pow(compFactor, Z.n) - 1) / (1 - discount);
}
double fundamentalValueOfAnnuityInArrear(Interest Z) {
  double discount = discountFactor(Z.i);
  return (1 - pow(discount, Z.n)) / Z.i;
}
double endValueOfAnnuityInArrear(Interest Z) {
  double compFactor = compoundingFactor(Z.i);
  return fundamentalValueOfAnnuityInArrear(Z) * pow(compFactor, Z.n);
}
