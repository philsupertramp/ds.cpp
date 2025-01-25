#include "../Test.h"
#include <math/numerics/analysis/SupportValues.h>
#include <math/visualization/Plot.h>


class PlotTestCase : public Test
{
  bool TestPlot() {
    Matrix<double> X = linspace(-5, 5, 5).Transpose();
    Matrix<double> Y = X.Apply([](const double& in) { return in * in; });

    LagrangeBase base(X, Y);
    Matrix<double> xi = linspace(-3.5, 3.5, 5).Transpose();
    auto res          = base.Evaluate(xi);

    Plot plot("Comparison of approximation with true value");

    plot.AddData(HorizontalConcat(X, Y), "support values");
    plot.AddData(HorizontalConcat(xi, res), "approximation");

    plot();

    return true;
  }

  bool TestScatterPlot() {
    Matrix<double> X = linspace(-5, 5, 5).Transpose();
    Matrix<double> Y = X.Apply([](const double& in) { return in * in; });
    ScatterPlot plot2("Test Scatter Plot: 2 Support Values");
    plot2.AddData(X, Y, "support values");
    plot2.pointSize(5);
    plot2.xAxis("X");
    plot2.yAxis("Y");
    plot2();

    return true;
  }
  bool TestFunctionPlot() {
    FunctionPlot plot3([](const double& in) { return in * in; }, "Function Plot: x^2");
    plot3.AddData(-5, 5, 0.1, "support values");
    plot3.xAxis("X");
    plot3.yAxis("Y");
    plot3();
    return true;
  }
  bool TestSurfacePlot() {
    SurfacePlot plot("Surface Plot Test");
    auto dat = ones(100, 3);
    plot.AddData(dat, "some data", DataTypes::LINE, nullptr, 3);
    plot();
    return true;
  }

  bool TestImagePlot() {
    ImagePlot plot("Image Plot: Test");
    auto in = zeros(10, 10);
    for(size_t i = 0; i < 10; ++i) {
      for(size_t j = 0; j < 10; ++j) {
        if(i <= 5 && j <= 5) { in(i, j) = 0.5; }
        if(i > 5 && j > 5) { in(i, j) = 1.0; }
      }
    }
    plot.AddData(in, "some data");
    plot();
    return true;
  }

public:
  virtual void run() {
    TestPlot();
    TestScatterPlot();
    TestFunctionPlot();
    TestImagePlot();
    TestSurfacePlot();
    TestImagePlot();
  }
};

int main() { PlotTestCase().run(); }
