#include "../Test.h"
#include <math/numerics/Fractals.h>
#include <math/numerics/utils.h>
#include <math/visualization/Plot.h>


class FractalsTestCase : public Test
{
  bool TestNewton() {
    NewtonFractal fractal(10);
    auto M = fractal();

    Matrix<double> expected({
    { 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2 },
    { 1, 1, 1, 3, 2, 2, 2, 2, 1, 3, 3, 2, 1, 1, 1, 1, 3, 2, 2, 2 },
    { 1, 2, 1, 3, 2, 2, 2, 1, 3, 3, 3, 3, 2, 1, 1, 1, 3, 2, 1, 2 },
    { 2, 2, 2, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 1, 1, 1 },
    { 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1 },
    { 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1 },
    { 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2 },
    { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
    { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
    });

    //        AssertEqual(M, expected);

    fractal = NewtonFractal(100);
#if USE_VIS
    ImagePlot plot("Newton-Fractals");
    plot.AddData(fractal(), "");
    plot("inferno.pal");
#endif
    return true;
  }

  bool TestMandelbrot() {
    Mandelbrot fractal;
    fractal.detail = 10;
    auto M         = fractal();

    Matrix<double> expected({ { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
                              { 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
                              { 2, 3, 4, 4, 5, 0, 5, 4, 4, 3 },
                              { 4, 4, 4, 5, 6, 0, 6, 5, 4, 4 },
                              { 4, 4, 5, 8, 0, 0, 0, 8, 5, 4 },
                              { 4, 5, 7, 9, 17, 0, 17, 9, 7, 5 },
                              { 5, 7, 27, 0, 0, 0, 0, 0, 27, 7 },
                              { 12, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                              { 4, 6, 16, 0, 0, 13, 0, 0, 16, 6 },
                              { 3, 4, 4, 4, 5, 5, 5, 4, 4, 4 } });

    AssertEqual(M, expected);

    fractal.detail   = 250;
    fractal.maxIters = 99;
#if USE_VIS
    ImagePlot plot("Mandelbrot");
    plot.AddData(fractal(), "");
    plot();
#endif

    return true;
  }

public:
  virtual void run() {
    TestNewton();
    TestMandelbrot();
  }
};

int main() {
  FractalsTestCase().run();
  return 0;
}
