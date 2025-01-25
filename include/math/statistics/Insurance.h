/**
 * @file Insurance.h
 *
 * Contains methods of the field of insurance mathematics.
 */
#pragma once
#include "Probability.h"


/**
 * Interest representation
 */
struct Interest {
  //! Interest rate
  double i;
  //! Period length
  double n;
  //! Starting balance
  double B_0;
  //! End balance
  double B_N;
  //! annual Pension
  double R;
};

/**
 * Compounding factor
 * $$1+i$$
 * @param i interest rate
 * @returns 1+i
 */
double compoundingFactor(double i);
/**
 * Discounting factor
 * $$\frac{1}{1+i}$$
 *
 * @param i interest rate
 * @returns 1/(1+i)
 */
double discountFactor(double i);

/**
 * calculates the fundamental value, the starting balance
 * @param I values to compute from
 * @returns fundamental balance
 */
double fundamentalValue(Interest I);
/**
 * calculates the end value, the balance at the end of I.n periods
 * @param I values to compute from
 * @returns
 */
double endValue(Interest I);
/**
 * calculates the term per period of given interest
 * @param I values to compute from
 * @returns
 */
double termInPeriods(Interest I);
/**
 * calculates the fundamental value, the starting balance
 * for annuity in advance.
 * @param I values to compute from
 * @returns
 */
double fundamentalValueOfAnnuityInAdvance(Interest I);
/**
 * calculates the end value, the balance at the end of I.n periods
 * for annuity in advance
 * @param I values to compute from
 * @returns
 */
double endValueOfAnnuityInAdvance(Interest I);
/**
 * calculates the fundamental value, the starting balance
 * for annuity in arrear.
 * @param I values to compute from
 * @returns
 */
double fundamentalValueOfAnnuityInArrear(Interest I);
/**
 * calculates the end value, the balance at the end of I.n periods
 * for annuity in arrear.
 * @param I values to compute from
 * @returns
 */
double endValueOfAnnuityInArrear(Interest I);

/**
 * \example statistics/TestInsurance.cpp
 * This is an example on how to use the Insurance.h file.
 */