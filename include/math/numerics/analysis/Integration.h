#pragma once
#include "../../Matrix.h"
#include "../../sorting.h"
#include "../utils.h"
#include <functional>

namespace integration {
  /**
   * Recursive implementation of numerical quadrature to approximate a given exact integral
   * uses Simspon's - and Trapezoid Rule for approximation.
   *
   * The Trapezoid Rule is given as
   *
   * \begin{equation}
   *  \int \limits_a^b f(x) dx
   *  \approx
   *  T(x) = \frac{b - a}{2}
   *  \left[
   *     f(a) + f(b)
   *  \right]
   * \end{equation}
   * Simpson is given as
   * \begin{equation}
   *  \int \limits_a^b f(x) dx
   *  \approx
   *  S(x) = \frac{
   *      b - a
   *  }{
   *      6
   *  }
   *  \left[
   *      f(a) + 4 f
   *      \left(
   *          \frac{a+b}{2}
   *      \right)
   *      + f(b)
   *  \right]
   * \end{equation}
   *
   * For approximation a tolerance is given, which is used to compare against the local error
   * of approximation. The error is calculated as
   * $$
   *  | S - T |
   * $$
   * Whenever the error reaches the threshold of tol, the method has succeeded and
   * the result of Simpson's rule is returned.
   *
   * **Note** if (b-a) < hMin we reached the minimal step with and haven't found
   * an satisfying result, therefore the algorithm exits with error code 1
   *
   * @param fun given Function to calculate the exact integral of
   * @param a left boundary of integral
   * @param b right boundary of integral
   * @param tol local tolerance
   * @param hMin minimum distance between a and b
   * @param nodes already visited nodes on the integral ("x-axis")
   * @param values already calculated function values for visited nodes
   * @returns approximation of given exact integral.
   */
  std::pair<double, std::vector<double>> quad_adaptive_rec(
  const std::function<double(double)>& fun,
  const double& a,
  const double& b,
  const double& tol,
  const double& hMin,
  std::vector<double>& nodes,
  Matrix<double>& values) {
    double h = b - a;
    if(h < hMin) {
      // error
      std::cerr << "H too small! h = " << h << std::endl;
      exit(-1);
    }

    Matrix<double> x = { { a }, { a + (h / 2.0) }, { b } };
    if(values.rows() == 0) {
      values = x.Apply(fun);
    } else if(values.rows() < 3) {
      values = { { values(0, 0) }, { fun(x(1, 0)) }, { values(1, 0) } };
    }

    auto qSimpson = (h / 6.0) * (values(2, 0) + 4.0 * values(1, 0) + values(0, 0));
    auto qTrapez  = (h / 2.0) * (values(2, 0) + values(0, 0));

    auto error = fabs(qSimpson - qTrapez);

    nodes.push_back(x(1, 0));

    if(error <= tol) { return { qSimpson, nodes }; }

    auto hTol        = tol / 2.0;
    auto leftValues  = values.GetSlice(0, 1, 0, 0);
    auto resLeft     = quad_adaptive_rec(fun, a, x(1, 0), hTol, hMin, nodes, leftValues);
    auto rightValues = values.GetSlice(1, 2, 0, 0);
    auto resRight    = quad_adaptive_rec(fun, x(1, 0), b, hTol, hMin, nodes, rightValues);

    return { resLeft.first + resRight.first, nodes };
  }
} // namespace integration

/**
 * Calculates the given exact integral
 *
 * \begin{equation}
 *  \int \limits_{l}^{u} f(x) dx
 * \end{equation}
 * where $l$ := lower boundary, and $u$ := upper boundary
 *
 *
 * @param fun Function to calculate exact integral of
 * @param lower lower boundary of integral
 * @param upper upper boundary of integral
 * @param tol tolerance of result to exact solution
 * @param hMin minimal distance between approximation points
 * @returns approximated exact integral of fun
 */
std::pair<double, std::vector<double>> quadrature(
const std::function<double(double)>& fun,
const double& lower,
const double& upper,
const double& tol,
const double& hMin) {
  std::vector<double> nodes = { lower, upper };
  Matrix<double> q;
  auto res   = integration::quad_adaptive_rec(fun, lower, upper, tol, hMin, nodes, q);
  res.second = sort(res.second);
  return res;
}
/**
 * \example numerics/analysis/TestIntegration.cpp
 * This is an example on how to use the quadrature method.
 */
