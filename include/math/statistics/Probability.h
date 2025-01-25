/**
 * @file Probability.h
 *
 * Contains methods of the field of probability calculation.
 */
#pragma once

#include "../Matrix.h"
#include "../matrix_utils.h"

/**
 * Linear Regression Model
 */
struct LinearModel {
  //! $$\beta_0$$
  double beta_0;
  //! $$\beta_1$$
  double beta_1;
  //! $$\hat{y}$$ approximated y value for given samples
  Matrix<double> y_estimate;
  //! $$\epsilon_i$$ Residuals of the underlying squared error computations
  Matrix<double> residuals;
};

/**
 * enum to chose a norm to compute
 */
enum P_NORM {
  //! $$||x||_\infty$$
  Inf,
  //! $$||x||_1$$
  One,
  //! $$||x||_2$$
  Eukl
};

/** regular helpers **/
/**
 * Method to calculate exponent of given value
 * @param x value to calculate exponent of
 * @returns exponent
 */
double getExponent(double x);
/**
 * Rounds double value to given precision, cuts off other elements
 * @param x value to round
 * @param precision precision in resulting value
 * @returns rounded value
 */
double round(double x, int precision);
/**
 * Absolute value $$|x|$$
 * @param x a value
 * @returns $$|x|$$
 */
double abs(double x);
/**
 * calculates $$x^{exponent}$$
 * @param x a value
 * @param exponent desired exponent
 * @returns $$x^{exponent}$$
 */
double pow(double x, int exponent);
/**
 * Calculates the Norm of a given vector
 * @param vec vector to calculate the norm of
 * @param pNorm desired norm to compute
 * @returns computed norm value
 */
double norm(const Matrix<double>& vec, P_NORM pNorm);
/**
 * Computes covariance of two given vectors
 * $$\text{cov}(x, y) = E[(x - E[x])(y - E[y])]$$
 *
 * The covariance is a measure of joint variability of two random variables.
 * - Positive covariance -> variables have same tendencies, i.e. high and low values in both variables at the same time
 * - Negative covariance -> variables have different tendencies, i.e. high values in variable $x$ correspond to low values in variable $y$ and vice versa
 *
 * @param x a vector
 * @param y another vector
 * @returns resulting covariance
 */
double cov(const Matrix<double>& x, const Matrix<double>& y);
/**
 * Computes variance of given vector
 * $$\text{var}(x)=E[(X - \mu)^2] = \text{cov}(X,X)$$
 *
 * The variance is the expectation of the square deviation of a random variable from its population mean or sample mean.
 *
 * @param x a vector
 * @returns $$\text{var}(x)$$
 */
double var(const Matrix<double>& x);

/**
 * Computes standard deviation of given vector
 * $$
 * \text{sd}(x) = \sigma(x) = \sqrt{E[(X-\mu)^2]} = \sqrt{E[X^2] - (E[X])^2}
 * $$
 *
 * The standard deviation is a measure of variance or dispersion of a set of values.
 *
 * @param x set of values to compute standard deviation from
 * @returnss standard deviation of given x
 */
Matrix<double> sd(const Matrix<double>& x, int axis = 0);
/**
 * Computes linear (regression) model for given random sample
 * $$
 * (Y_i; X_i)
 * $$
 * $$
 * Y_i = \beta_0 + \beta_1\phi_1(x_i) + \epsilon_i
 * $$
 * @param x $$X_i$$
 * @param y $$Y_i$$
 * @returns constructed linear model
 */
LinearModel lm(const Matrix<double>& x, const Matrix<double>& y);
/**
 * The coefficient of determination ($R^2$) is the proportion of the variation in the dependent variable that is
 * predictable from independent variables.
 *
 *$$R^2 = \frac{\text{var}(y)}{\text{var}(\hat{y})}$$
 * @param y $$y$$
 * @param yHat $$\hat{y}$$
 * @returns proportion of the variation
 */
double coefficientOfDetermination(const Matrix<double>& y, const Matrix<double>& yHat);
/**
 * Computes likelihood
 * @returns
 */
double likelihood(const Matrix<double>&);

/**
 * Computes orthogonal-linear regression for given values A, mapped onto
 * points $$P(i, a_i)$$.
 *
 * $$Y_i = \beta_0 + \beta_1 * a_i$$
 *
 * @param A vector of y-values
 * @returns A vector of x-value approximations
 */
Matrix<double> Regression(const Matrix<double>& A);

/**
 * Computes correlation matrix of two matrices
 *
 * $$
 * \text{corr}(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X\sigma_Y}
 * $$
 * with $\sigma$ the standard deviations of $X$ and $Y$.
 *
 * The correlation matrix contains correlation coefficients $\text{corr}_{x_i, y_i} \in [-1, 1]$.
 * - negative correlation coefficients: The values negatively influence each other
 * - positive correlation coefficients: The values positively influence each other
 * - neutral correlation coefficients: The values don't influence each other
 * @param A the first matrix of values
 * @param B second matrix of values in same shape as A
 * @returns The correlation matrix of A and B
 */
Matrix<double> corr(const Matrix<double>& A, const Matrix<double>& B);

/**
 * Computes the expected value of a given vector.
 *
 * $$
 * E[X] = \sum x_i P(x_i)
 * $$
 *
 * The expected value is a generalization of the weighted average.
 * Given a set of random values, the expected value is the value that is most likely to occur within given distribution of values.
 *
 * @param A the matrix to caclulate the expected value(s) of, each row will receive a corresponding expected value
 * @returns The expected value(s) of A
 */
Matrix<double> expected_value(const Matrix<double>& A);

/**
 * \example statistics/TestProbability.cpp
 * This is an example on how to use the Porbability.h file.
 */
