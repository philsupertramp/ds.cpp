#pragma once

#include "../Matrix.h"
#include "../matrix_utils.h"
#include "../numerics/utils.h"
#include <utility>


/**
 * data type enum to display a specific dataset
 */
enum DataTypes {
  //! Creates line data
  LINE = 1,
  //! Creates dot data
  DOTS = 2,
  //! Unknown data type
  NONE = -1,
};

/**
 * Getter for name strings of types
 * @param type type to get string representation of
 * @returns string representation of type
 */
const char* GetPlotDataTypeName(DataTypes type) {
  switch(type) {
    case LINE: return "lines ls 1";
    case DOTS: return "points ls 2"; // pointtype 5 pointsize 1.5";
    case NONE:
    default: return "none";
  }
}

/**
 * representation of min-max boundaries for a axis
 */
struct PlotBoundary {
  //! min boundary value
  double min;
  //! max boundary value
  double max;
};

/**
 * Index representation of a plot dataset
 */
struct PlotIndex {
  //! start index
  int start = -1;
  //! end index
  int stop = -1;
};

/**
 * struct representing attributes of a plot
 */
struct PlotAttributes {
  //! title label
  char* title;
  //! x-axis label
  char* xAxis;
  //! y-axis label
  char* yAxis;
  //! point strength
  int pointStrength;
  //! line strength
  int lineStrength;
  //! signalizes 3D plot
  bool is3D = false;
  //! signalizes 2D plot
  bool is2D = false;
  //! identifier for statistical plots for value comparison
  unsigned int isStatPlot;
  //! color used to plot elements
  const char* color;
  //! color used to plot elements
  const char* colorPalette;
  //! vector representing data series
  std::vector<const char*> plotNames;
  //! vector representing data series types
  std::vector<const char*> plotTypes;
  //! vector representing characters to use to display points per data series
  std::vector<const char*> characters;
  //! vector containing dataset indices in the storage file
  std::vector<PlotIndex> plotIndices;
};

/**
 * Represents a data plot, using gnuplot.
 *
 * Add data using the AddData methods. After you are done, call the plot to display the results.
 *
 *
 * ===============
 * Examples:
 *  \code
 *  Plot plot("My plot");
 *  plot.AddData({{1, 2}, {2, 3}}, "Some points", DataType::POINT);
 *  plot();
 *  \endcode
 */
class Plot
{
public:
  /**
   * default constructor.
   * Prepares context for adding data.
   * @param title desired title for plot
   */
  explicit Plot(const std::string& title = "") {
    dataFileName             = "line_plot_data.txt";
    plotTypeName             = "lines";
    attributes.pointStrength = 0;
    attributes.lineStrength  = 10;
    attributes.isStatPlot    = 0;
    attributes.color         = "#008080";
    attributes.title         = new char[title.size() + 1];
    std::copy(title.begin(), title.end(), attributes.title);
    attributes.title[title.size()] = '\0';
    clear_file();
  }

  void clear_file() {
    // create file and empty it
    storageFile = fopen(dataFileName, "w");
    fprintf(storageFile, "\0");
    fclose(storageFile);
  }

  /**
   * Setter for title label
   * @param title desired plot title
   */
  void title(const std::string& title) {
    attributes.title = new char[title.size() + 1];
    std::copy(title.begin(), title.end(), attributes.title);
    attributes.title[title.size()] = '\0';
  }

  /**
   * Setter for x-axis label
   * @param name desired label name
   */
  void xAxis(const std::string& name) {
    attributes.xAxis = new char[name.size() + 1];
    std::copy(name.begin(), name.end(), attributes.xAxis);
    attributes.xAxis[name.size()] = '\0';
  }

  /**
   * Setter for y-axis label
   * @param name desired label name
   */
  void yAxis(const std::string& name) {
    attributes.yAxis = new char[name.size() + 1];
    std::copy(name.begin(), name.end(), attributes.yAxis);
    attributes.yAxis[name.size()] = '\0';
  }

  /**
   * setter for point size of plot data
   * @param size desired point size
   */
  void pointSize(const int& size) { attributes.pointStrength = size; }

  /**
   * setter for line width of plot data
   * @param size desired line size
   */
  void lineWidth(const int& size) { attributes.lineStrength = size; }

  /**
   * Helper to add pair-wise elements to plot context
   *
   * concatenates given X and Y and forwards new matrix to AddData with single matrix argument
   * @param x x values
   * @param y y values
   * @param name name of dataset
   * @param dataType plot data type
   */
  void AddData(
  const Matrix<double>& x,
  const Matrix<double>& y,
  const std::string& name,
  DataTypes dataType    = DataTypes::NONE,
  const char* character = nullptr) {
    x.assertSize(y);
    AddData(HorizontalConcat(x, y), name, dataType, character);
  }

  /**
   * Appends data to the storage file
   * @param mat matrix containing dataset
   * @param name name of data set
   * @param dataTypeName desired plot type
   * @param dimensions indicator for dimension of dataset
   */
  void AddData(
  const Matrix<double>& mat,
  const std::string& name,
  DataTypes dataTypeName = DataTypes::NONE,
  const char* pointChar  = nullptr,
  int dimensions         = 2) {
    if(dimensions <= 3) {
      // calculate boundaries
      auto new_bX = getBoundaries(mat.GetSlice(0, mat.rows() - 1, 0, 0));
      auto new_bY = getBoundaries(mat.GetSlice(0, mat.rows() - 1, 1, 1));

      bX.min = bX.min > new_bX.min ? new_bX.min : bX.min;
      bX.max = bX.max < new_bX.max ? new_bX.max : bX.max;

      bY.min = bY.min > new_bY.min ? new_bY.min : bY.min;
      bY.max = bY.max < new_bY.max ? new_bY.max : bY.max;
    } else {
      bX.min = 0;
      bY.min = 0;
      bX.max = mat.columns();
      bY.max = mat.columns();
    }

    // append storage file
    storageFile = fopen(dataFileName, "a");
    if(storageFile == nullptr) {
      // error
      return;
    }
    fprintf(storageFile, "# ");
    fprintf(storageFile, "%s", dataFileName);
    fprintf(storageFile, "\n");

    for(size_t i = 0; i < mat.columns(); i += dimensions) {
      fprintf(storageFile, "# X Y Z\n");
      for(size_t j = 0; j < mat.rows(); ++j) {
        for(int k = 0; k < dimensions; ++k) { fprintf(storageFile, "%g ", mat(j, i + k)); }
        fprintf(storageFile, "\n");
      }
      fprintf(storageFile, "\n\n");
    }
    fclose(storageFile);

    char* newName;
    newName = new char[name.size() + 1];
    std::copy(name.begin(), name.end(), newName);
    newName[name.size()] = '\0';
    PlotIndex index      = { 0, -1 };
    if(!attributes.plotIndices.empty()) {
      index = attributes.plotIndices[attributes.plotIndices.size() - 1];
      index.start++;
    }

    attributes.plotIndices.push_back(index);
    attributes.plotNames.push_back(newName);
    attributes.plotTypes.push_back(dataTypeName == DataTypes::NONE ? plotTypeName : GetPlotDataTypeName(dataTypeName));
    attributes.characters.push_back(pointChar);
    numElements++;
  }

  /**
   * execution using call operator
   *
   * Forwards data to gnuplot
   */
  virtual void operator()() const {
    FILE* gnuplot = popen("gnuplot --persist", "w");
    writeAttributes(gnuplot);
    for(int i = 0; i < numElements; ++i) {
      if(i == 0) { fprintf(gnuplot, "%s", plotType); }
      fprintf(gnuplot, "'%s'", dataFileName);
      auto index = attributes.plotIndices[i];
      if(index.stop != -1) {
        fprintf(gnuplot, " index %d:%d with ", index.start, index.stop);
      } else {
        fprintf(gnuplot, " index %d with ", index.start);
      }
      fprintf(gnuplot, "%s", attributes.plotTypes[i]);
      if(attributes.characters[i] != nullptr) { fprintf(gnuplot, "pt '%s'", attributes.characters[i]); }
      fprintf(gnuplot, " title '%s'", attributes.plotNames[i]);
      fprintf(gnuplot, i == (numElements - 1) ? "\n" : ",");
    }
    fprintf(gnuplot, "\n");
    fclose(gnuplot);
  }

private:
  /**
   * calculates boundaries of given matrix (min and max values)
   * @param x input vector to calculate boundaries of
   * @returns boundaries of input
   */
  [[nodiscard]] inline PlotBoundary getBoundaries(const Matrix<double>& x) const {
    /*
     * determines the boundaries of an array
     * returns: boundary(min, max)
     */
    double startX = min(x);
    //startX = startX - pow_(10, getExponent(startX) - 1);
    double endX = max(x);
    //endX = endX - pow_(10, getExponent(endX) - 1);
    PlotBoundary bond = { startX, endX };
    return bond;
  }


protected:
  //! data storage file name
  const char* dataFileName;
  //! representation of plot type
  const char* plotTypeName;
  //! representation of plot type
  const char* plotType = "plot";
  //! data storage file
  FILE* storageFile = nullptr;
  //! attributes for plot
  PlotAttributes attributes = {};
  //! number elements within the plot
  int numElements = 0;
  //! x-axis boundaries for the resulting plot
  PlotBoundary bY{};
  //! y-axis boundaries for the resulting plot
  PlotBoundary bX{};

  /**
   * helper to write attributes to gnuplot context
   * @param gnuplot pointer to gnuplot stream
   */
  void writeAttributes(FILE* gnuplot) const {
    if(attributes.title) {
      char titleExtension[105];
      if(attributes.xAxis && attributes.yAxis && attributes.isStatPlot) {
        strcpy(titleExtension, attributes.xAxis);
        strcat(strcat(titleExtension, " vs. "), attributes.yAxis);
      }
      fprintf(gnuplot, "set title '%s", attributes.title);
      if(attributes.isStatPlot) { fprintf(gnuplot, "%s", titleExtension); }
      fprintf(gnuplot, "'\n");
    }
    fprintf(gnuplot, "set style line 1 lt 2 lc rgb \"red\" lw 3;\nset style line 1 lt 2 lc rgb \"orange\" lw 3;\n");
    if(attributes.xAxis) { fprintf(gnuplot, "set xlabel '%s'\n", attributes.xAxis); }
    if(attributes.yAxis) { fprintf(gnuplot, "set ylabel '%s'\n", attributes.yAxis); }
    if(attributes.is3D) { fprintf(gnuplot, "set hidden3d\nset dgrid3d 30,30 qnorm 1\n"); }
    if(attributes.colorPalette) { fprintf(gnuplot, "load '%s'\n", attributes.colorPalette); }
  }
};

/**
 * Wrapper class to display a scatter plot.
 */
class ScatterPlot : public Plot
{
public:
  /**
   * default constructor
   * @param name plot title
   */
  explicit ScatterPlot(const char* name = "")
    : Plot(name) {
    dataFileName = "point_plot_data.txt";
    plotTypeName = "points";
    clear_file();
  }
};

/**
 * Wrapper class to easily generate a plot based on a function.
 */
class FunctionPlot : public Plot
{
  //! Function to evaluate and display
  std::function<double(const double&)> Function;

public:
  /**
   * default constructor, requires function fun
   * @param fun function to plot
   * @param name plot title
   */
  explicit FunctionPlot(std::function<double(const double&)> fun, const char* name = "")
    : Plot(name)
    , Function(std::move(fun)) { }

  /**
   * Adds data to the plot context
   * @param start starting value
   * @param end end value
   * @param stepSize desired step size
   * @param name data set name
   */
  void AddData(const double& start, const double& end, const double& stepSize, const char* name) {
    int size         = (int)((end - start) / stepSize) + 2;
    Matrix<double> X = linspace(start, end, size);
    auto Y           = X.Apply(Function);
    Plot::AddData(X, Y, name);
  }
};

/**
 * Plot for only point data
 */
class SurfacePlot : public Plot
{
public:
  /**
   * Default constructor
   * @param title plot title
   */
  SurfacePlot(const std::string& title = "")
    : Plot(title) {
    plotType        = "splot";
    attributes.is3D = true;
  }

  /**
   * overridden call operator.
   *
   * interprets a file as surface plot and forwards whole content to gnuplot using splot
   */
  void operator()() const override {
    FILE* gnuplot = popen("gnuplot --persist", "w");
    writeAttributes(gnuplot);
    fprintf(gnuplot, "%s", plotType);
    fprintf(gnuplot, " '%s' with lines", dataFileName);
    fprintf(gnuplot, " title '%s'", attributes.plotNames[0]);
    fprintf(gnuplot, "\n");
    fclose(gnuplot);
  }
};

/**
 * Plot to display a 2 dimensional image
 * based on matrix-like data
 */
class ImagePlot : public Plot
{
public:
  /**
   * Default constructor
   *
   * @param name Plot title
   */
  explicit ImagePlot(const char* name)
    : Plot(name) {
    attributes.is2D = true;
  }

  /**
   * Helper to easily add matrix data to content file
   * @param mat data matrix
   * @param name name of dataset
   */
  void AddData(const Matrix<double>& mat, const std::string& name) {
    Plot::AddData(mat, name, DataTypes::NONE, nullptr, mat.columns());
  }

  /**
   * overridden call operator.
   *
   * interprets a file as surface plot and forwards whole content to gnuplot using "plot matrix with image"
   */
  void operator()(const char* colorPalette = "magma.pal") {
    FILE* gnuplot           = popen("gnuplot --persist", "w");
    attributes.colorPalette = colorPalette;
    writeAttributes(gnuplot);
    fprintf(gnuplot, "plot '%s' matrix with image", dataFileName);
    fprintf(gnuplot, " title '%s'", attributes.plotNames[0]);
    fprintf(gnuplot, "\n");
    fclose(gnuplot);
  }
};
/**
 * \example plot/TestPlot.cpp
 * This is an example on how to use the Plot, ScatterPlot, FunctionPlot and SurfacePlot classes.
 */
