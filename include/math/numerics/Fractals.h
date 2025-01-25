/**
 * @file math/numerics/Fractals.h
 *
 *
 * Requires:
 * \code
 * #include <math/numerics/Fractals.h>
 * \endcode
 *
 * This file contains several implementations of fractals using different
 * approaches to generate a somewhat "nice" visualization.
 *
 * Printing the result matrix can already look quite nice!
 *
 * \code
 * NewtonFractal fractal(25);
 * fractal.maxIter = 2000;
 * fractal.tol = 1e-13;
 * fractal();
 * >>> [
 *    	0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
 *    	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2, 2, 2,
 * 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 2, 2, 2, 2, 3, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2,
 * 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
 *    	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
 * 3,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2,
 * 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
 * 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
 *    	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 1,
 * 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 1,
 * 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3,
 * 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1,
 * 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3
 *    	1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 * 1,
 * 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2,
 * 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3,
 * 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1,
 * 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
 * 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 1, 1,
 * 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3 2, 2, 1, 1, 2, 2, 1, 1, 3, 3, 1,
 * 2, 2, 2, 1, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 1, 3, 2, 2, 2, 3, 1, 1, 3, 3, 2, 2, 3, 3, 2, 2
 *    	3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 1, 2, 3,
 * 3,
 * 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 3, 3, 3, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 3, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1 3, 3, 3, 3, 3,
 * 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
 * 1, 1, 1, 1, 1, 1 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1 3, 3, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 1, 1, 1, 1 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1
 *    	3, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2,
 * 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 1 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3 3, 1, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 3, 1 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
 *    	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
 * 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
 * ]// clang-format on
 * \endcode
 *
 * or as plot using ImagePlot
 *
 * \image html newton-fractal.svg width=50%
 *
 * \code
 * Mandelbrot fractal;
 * fractal.detail   = 1250;
 * fractal.maxIters = 99;
 * ImagePlot plot("Mandelbrot");
 * plot.AddData(fractal(), "");
 * plot();
 * \endcode
 * \image html mandelbrot.svg width=50%
 */
#pragma once

#include "lin_alg/newton.h"


/**
 * Newton fractals
 */
class NewtonFractal
{
public:
  //! detail factor for approximation
  double detail;
  //! start and end value on x-axis
  double xMin = -1.0, xMax = 1.0;
  //! start and end value on y-axis
  double yMin = -1.0, yMax = 1.0;
  //! max number iterations for newton algorithm
  int maxIter = 100;
  //! tolerance used for newton algorithm
  double tol = 1e-5;

  /**
   * Default constructor
   * @param detailFactor determines how detailed the fractal is displayed
   * @param _min min value (chose from [-1, 1])
   * @param _max max value (chose from [-1, 1])
   * @param _maxIter number maximum iterations for newton algorithm
   * @param _tol tolerance used for newton algoirthm
   */
  NewtonFractal(double detailFactor = 1, double _min = -1.0, double _max = 1.0, int _maxIter = 100, double _tol = 1e-5)
    : detail(detailFactor)
    , xMin(_min)
    , xMax(_max)
    , yMin(_min)
    , yMax(_max)
    , maxIter(_maxIter)
    , tol(_tol) { }

  /**
   * Approximates newton fractal given by NewtonFractal::fun and NewtonFractal::jac.
   *
   * @returns Classified roots, meaning pixels approximated with same roots hold same values
   */
  Matrix<double> operator()() const {
    size_t n = fabs(xMax - xMin) / (1 / (detail > 10 ? detail : 10));
    auto x_i = linspace(xMin, xMax, n).Transpose();
    auto y_i = linspace(yMin, yMax, n).Transpose();

    auto M = zeros(n, n);
    std::vector<Matrix<double>> roots;

    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j < n; ++j) {
        auto res = newton(fun, jac, { { x_i(i, 0) }, { y_i(j, 0) } }, tol, maxIter);
        if(res.second == maxIter) {
          M(i, j) = 0;
        } else {
          size_t index = 0;
          for(size_t k = 0; k < roots.size(); ++k) {
            if(norm(res.first - roots[k]) <= tol) { index = k; }
          }
          if(index == 0) {
            index = roots.size();
            roots.push_back(res.first);
          }
          M(i, j) = (double)index;
        }
      }
    }
    return M;
  }

private:
  /**
   * Newton fractal function
   *
   * $$
   * \begin{pmatrix}
   * x_0^3 - 3x_0 \cdot x_1^2 - 1 \\\\
   * -x_1^3 + 3x_0^2 \cdot x_1
   * \end{pmatrix}
   * $$
   * @param x point to evaluate
   * @returns evaluated value
   */
  static Matrix<double> fun(const Matrix<double>& x) {
    return Matrix<double>({ { pow(x(0, 0), 3) - 3 * x(0, 0) * x(1, 0) * x(1, 0) - 1 },
                            { -pow(x(1, 0), 3) + 3 * x(0, 0) * x(0, 0) * x(1, 0) } });
  }
  /**
   * Helper, jacobian matrix of fun
   *
   * $$
   * \begin{pmatrix}
   * 3 x_0^2 x_1^2 & -6 x_0x_1\\\\
   * 6 x_0 x_1 & 3 x_0^2 - 3x_1^2
   * \end{pmatrix}
   * $$
   * @param x current value (2 dimensional)
   * @returns 2 by 2 jacobian matrix evaluated in x
   */
  static Matrix<double> jac(const Matrix<double>& x) {
    return Matrix<double>({ { 3 * (x(0, 0) * x(0, 0) - x(1, 0) * x(1, 0)), -6 * x(0, 0) * x(1, 0) },
                            { 6 * x(0, 0) * x(1, 0), 3 * x(0, 0) * x(0, 0) - 3 * x(1, 0) * x(1, 0) } });
  }
};

/**
 * Mandelbrot fractal
 */
class Mandelbrot
{
public:
  //! max number iterations for approximation of single value
  size_t maxIters = 125;
  //! start and end value on x-axis
  double startX = -2.5, endX = 1.0;
  //! start and end value on y-axis
  double startY = -1.0, endY = 1.0;
  //! detail factor for approximation
  size_t detail = 500;


  /**
   * default constructor
   */
  Mandelbrot() { }


  /**
   * Evaluates mandelbrot for given view.
   * @returns
   */
  Matrix<double> operator()() {
    auto stepWidthX = (endX - startX) / (double)detail;
    auto stepWidthY = (endY - startY) / (double)detail;

    Matrix<double> M = zeros(detail, detail);
    for(size_t x = 0; x < detail; ++x) {
      for(size_t y = 0; y < detail; ++y) {
        M(x, y) = (double)fun((startX + stepWidthX * (double)x), (startY + stepWidthY * (double)y));
        if(M(x, y) == (double)maxIters) { M(x, y) = 0; }
      }
    }
    return M;
  }

private:
  /**
   * Evaluates mandelbrot at given point
   * @param real_c real part of C (chose from [-2.5, 1])
   * @param img_c imaginary part of C (chose from [-1, 1])
   * @returns iterations needed to evaluate for given values
   */
  size_t fun(const double& real_c, const double& img_c) {
    size_t iters = 0;
    double real  = 0;
    double img   = 0;

    while(++iters < maxIters && real * real + img * img < 4) {
      double temp = real;
      real        = real * real + real_c - img * img;
      img         = 2 * img * temp + img_c;
    }

    return iters;
  }
};
/**
 * \example numerics/TestFractals.cpp
 * This is an example on how to use the different Fractal classes.
 */