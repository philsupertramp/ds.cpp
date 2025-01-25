#pragma once

#include "../Matrix.h"
#include "../format.h"
#include "../matrix_utils.h"
#include <fstream>
#include <map>
#include <sstream>

/**
 * Flag map for normalizer methods
 */
enum NormalizerMethod { SET_MEAN = 0, COL_MEAN = 1, ROW_MEAN = 2 };

/**
 * Representation of
 */
struct Set {
  //! input data
  Matrix<double> Input;
  //! expected output data
  Matrix<double> Output;
  //! number of input elements
  size_t InputCount = 0;
  //! number of output elements
  size_t OutputCount = 0;
  //! number of input-output pairs
  size_t count = 0;

  /**
   * default constructor
   */
  Set() = default;

  /**
   * initialization constructor, just sets input and output dimensions
   * @param inputCount number input elements
   * @param outputCount number output elements
   */
  Set(size_t inputCount, size_t outputCount) {
    InputCount  = inputCount;
    OutputCount = outputCount;
  }

  /**
   * Constructor which immediately reads a given set file
   * @param fileName file name to read from
   *@param inputCount number input elements
   * @param outputCount number output elements
   */
  Set(const char* fileName, size_t inputCount, size_t outputCount) {
    InputCount  = inputCount;
    OutputCount = outputCount;
    ReadFromFile(fileName);
  }

  /**
   * reads content from file
   *
   * File needs to have tab separated records
   *
   *
   * @param fileName
   */
  void ReadFromFile(const char* fileName) {
    std::string line;
    size_t lineCount = 0;

    std::ifstream dataFile2(fileName);
    if(dataFile2.is_open()) {
      while(getline(dataFile2, line, '\n')) { lineCount++; }
      dataFile2.close();
    }
    std::ifstream dataFile(fileName);
    count  = 0;
    Input  = Matrix<double>(0, lineCount, InputCount);
    Output = Matrix<double>(0, lineCount, OutputCount);
    if(dataFile.is_open()) {
      line = "";
      while(getline(dataFile, line, '\n')) {
        std::string val;
        std::stringstream lineStream(line);
        for(size_t i = 0; i < InputCount + OutputCount; i++) {
          if(i < InputCount + OutputCount - 1) {
            std::getline(lineStream, val, '\t');
          } else {
            std::getline(lineStream, val);
          }
          if(i < InputCount) {
            Input(count, i) = std::atof(val.c_str());
          } else {
            Output(count, i - InputCount) = std::atof(val.c_str());
          }
        }
        ++count;
      }
      dataFile.close();
      //            Classes = OutputToClass(Output);
    } else {
      std::cerr << "DataSet::ReadFromFile: Unable to open file " << fileName << std::endl;
    }
  }

  /**
   * Generates batch of given size based on inputs and outputs
   * @param batchSize number elements per batch
   * @returns
   */
  [[nodiscard]] Set GetBatch(int batchSize) const {
    if(batchSize == -1 || batchSize == (int)count) { return *this; }

    auto newDS   = Set(InputCount, OutputCount);
    newDS.Input  = Matrix<double>(0, batchSize, InputCount);
    newDS.Output = Matrix<double>(0, batchSize, OutputCount);
    for(int i = 0; i < batchSize; i++) {
      int index = (int)(Random::Get() * count);
      for(size_t in = 0; in < InputCount; in++) { newDS.Input(i, in) = Input(index, in); }
      for(size_t out = 0; out < OutputCount; out++) { newDS.Output(i, out) = Output(index, out); }
    }
    return newDS;
  }

  /**
   * Normalizes set by computing mean over all elements
   */
  void NormalizeSetMean() {
    auto maxIndex = argmax(Input);
    auto minIndex = argmin(Input);
    auto maxVal   = Input(int(maxIndex / Input.columns()), int(maxIndex % Input.columns()));
    auto minVal   = Input(int(minIndex / Input.columns()), int(minIndex % Input.columns()));

    auto minMax = maxVal - minVal;
    for(size_t i = 0; i < Input.columns(); i++) {
      for(size_t j = 0; j < Input.rows(); j++) { Input(i, j) = (Input(i, j) - minVal) / minMax; }
    }
  }

  /**
   * Normalizes set by computing mean over rows
   */
  void NormalizeRowMean() {
    auto means = Matrix<double>(0, Input.rows(), 1);
    for(size_t i = 0; i < Input.rows(); ++i) {
      for(size_t j = 0; j < Input.columns(); ++j) { means(i, 0) += Input(i, j); }
      means(i, 0) /= Input.columns();
    }
    auto stds = Matrix<double>(0, Input.rows(), 1);
    for(size_t i = 0; i < Input.rows(); ++i) {
      for(size_t j = 0; j < Input.columns(); ++j) { stds(i, 0) += pow(Input(i, j) - means(i, 0), 2); }
      stds(i, 0) /= Input.columns();
    }
    for(size_t i = 0; i < Input.rows(); ++i) {
      for(size_t j = 0; j < Input.columns(); ++j) { Input(i, j) = (Input(i, j) - means(i, 0)) / stds(i, 0); }
    }
  }

  /**
   * Normalizes set by computing mean over columns
   */
  void NormalizeColMean() {
    Input      = Input.Transpose();
    auto means = Matrix<double>(0, Input.rows(), 1);
    for(size_t i = 0; i < Input.rows(); ++i) {
      for(size_t j = 0; j < Input.columns(); ++j) { means(i, 0) += Input(i, j); }
      means(i, 0) /= Input.columns();
    }
    auto stds = Matrix<double>(0, Input.rows(), 1);
    for(size_t i = 0; i < Input.rows(); ++i) {
      for(size_t j = 0; j < Input.columns(); ++j) { stds(i, 0) += pow(Input(i, j) - means(i, 0), 2); }
      stds(i, 0) /= Input.columns();
    }
    for(size_t i = 0; i < Input.rows(); ++i) {
      for(size_t j = 0; j < Input.columns(); ++j) { Input(i, j) = (Input(i, j) - means(i, 0)) / stds(i, 0); }
    }
    Input = Input.Transpose();
  }

  /**
   * helper to normalize set based on NormalizerMethod
   * @param method representation of normalization method
   */
  void Normalize(NormalizerMethod method = NormalizerMethod::SET_MEAN) {
    switch(method) {
      case SET_MEAN: NormalizeSetMean(); break;
      case COL_MEAN: NormalizeColMean(); break;
      case ROW_MEAN: NormalizeRowMean(); break;
    }
  }
};

/**
 * dataset representation
 */
class DataSet
{
  friend class ImageDataSet;

public:
  /**
   * regular constructor
   * @param filePath path with dataset data files
   * @param inputCount number input elements
   * @param outputCount number output elements
   */
  DataSet(const char* filePath, size_t inputCount, size_t outputCount)
    : InputCount(inputCount)
    , OutputCount(outputCount)
    , Training(format("%s%s", filePath, "training.dat").c_str(), inputCount, outputCount)
    , Validation(format("%s%s", filePath, "validation.dat").c_str(), inputCount, outputCount)
    , Test(format("%s%s", filePath, "test.dat").c_str(), inputCount, outputCount) { }

  /**
   * initialization constructor, initializes required structures by given dimensions
   * @param inputCount number input elements
   * @param outputCount number output elements
   */
  DataSet(size_t inputCount, size_t outputCount)
    : InputCount(inputCount)
    , OutputCount(outputCount)
    , Training(inputCount, outputCount)
    , Validation(inputCount, outputCount)
    , Test(inputCount, outputCount) { }

  /**
   * interface definition, does nothing here
   * @param filePath
   */
  virtual void PrepareDirectory([[maybe_unused]] const char* filePath) { }

public:
  //! number input elements
  size_t InputCount;
  //! number output elements
  size_t OutputCount;

  //! set for training
  Set Training;
  //! set for validation of training
  Set Validation;
  //! set to test after training
  Set Test;

  //! number of epochs while training
  int maxEpoch = 1000;
  //! threshold for loss to prevent over-fitting
  double stopThreshold = 0.001;
  //! learning rate
  double eta = 0.0051;
  //! number of elements per batch
  int batchSize = 5;
  //! use verbose output during fitting
  bool verbose = false;
};
