#pragma once
#include "Random.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <type_traits>

#include <signal.h>


/**
 * holds dimensions of a Matrix
 */
struct MatrixDimension {
  //! number rows
  size_t rows;
  //! number columns
  size_t columns;
  //! number elements
  size_t elemDim = 1;
};
/**
 * Represents a multi dimensional Matrix of data with type `T`
 *
 * The class is capable of regular matrix operations, including matrix - vector calculation.
 * It also holds several helper methods to calculate data science specific products or other
 * common operations.
 * @tparam T
 *
 * Element access using **matrix(row, column, element)** notation
 *
 * Accessed elements can be modified!
 */
template<typename T = double>
class Matrix
{
public:
  /**
   * Default constructor
   * @param val default value for all elements
   * @param rowCount number of rows
   * @param colCount number of columns
   * @param elementDimension number of dimensions per element
   */
  explicit Matrix(T val, size_t rowCount, size_t colCount, size_t elementDimension = 1) {
    Resize(rowCount, colCount, elementDimension);
    for(size_t i = 0; i < _rows * _columns * _element_size; ++i) { _data[i] = val; }
  }

  /**
   * Vector-assignment, assigns colCount values of type T
   * @param val pointer representation of array
   * @param colCount number of elements within val
   */
  explicit Matrix(T* val, size_t colCount) {
    Resize(1, colCount);
    for(size_t j = 0; j < _columns; j++) { _data[GetIndex(0, j)] = val[j]; }
  }

  /**
   * overwritten default constructor
   */
  Matrix() { }

  /**
   * Constructor using initializer_list's
   * @param lst
   */
  Matrix(const std::initializer_list<std::initializer_list<T>>& lst) {
    int i = 0, j = 0;
    auto rows = lst.size();
    auto cols = lst.begin()->size();
    Resize(rows, cols);
    for(const auto& l : lst) {
      for(const auto& v : l) {
        _data[GetIndex(i, j)] = v;
        ++j;
      }
      j = 0;
      ++i;
    }
  }

  /**
   * Constructor using multi-dimensional initializer_list's
   * @param lst
   */
  Matrix(const std::initializer_list<std::initializer_list<std::initializer_list<T>>>& lst) {
    int i = 0, j = 0, c = 0;
    auto rows  = lst.size();
    auto cols  = lst.begin()->size();
    auto elems = lst.begin()->begin()->size();
    Resize(rows, cols, elems);
    for(const auto& l : lst) {
      for(const auto& v : l) {
        for(const auto k : v) {
          _data[GetIndex(i, j, c)] = k;
          ++c;
        }
        c = 0;
        ++j;
      }
      j = 0;
      ++i;
    }
  }

  /**
   * Deep copy constructor
   * @param other
   */
  Matrix(Matrix const& other) {
    Resize(other._rows, other._columns, other._element_size);
    std::memcpy(this->_data, other._data, other._dataSize * sizeof(T));
    this->needsFree = true;
  };

  /**
   * Conversion constructor to convert Matrix into other type V
   *
   * @param other the matrix to use
   * @returns given matrix casted to type V
   */
  template<typename V>
  Matrix(const Matrix<V>& other) {
    Resize(other.rows(), other.columns(), other.elements());
    auto index_factor = _columns * _element_size;
    for(size_t i = 0; i < (_rows * _columns * _element_size); ++i) {
      _data[i] = static_cast<T>(other(i / (index_factor), i % (index_factor), i % _element_size));
    }

    this->needsFree = true;
  }

  /**
   * Default destructor, doesn't do anything
   */
  ~Matrix() {
    if(needsFree) { free(_data); }
  }

  /**
   * Generates a random matrix
   * @param rows number of rows in target matrix
   * @param columns number of columns in target matrix
   * @param element_size
   * @param minValue
   * @param maxValue
   * @returns matrix of dimension `rows`, `columns` initialized with random values from `minValue` to `maxValue`
   */
  static Matrix
  Random(size_t rows, size_t columns, size_t element_size = 1, double minValue = 0.0, double maxValue = 1.0) {
    Matrix<T> matrix(0, rows, columns, element_size);
    for(size_t i = 0; i < rows * columns * element_size; ++i) { matrix._data[i] = Random::Get(minValue, maxValue); }
    return matrix;
  }

  /**
   * Generates normal distributed squared matrix
   *
   * @param rows
   * @param columns
   * @returns data set of normal distributed data
   */
  static Matrix Normal(size_t rows, size_t columns, double mu, double sigma) {
    assert(columns % 2 == 0);

    constexpr double two_pi = 2.0 * M_PI;
    Matrix out;
    out.Resize(rows, columns);
    for(size_t i = 0; i < rows * columns; i += 2) {
      auto u1 = Random::Get();
      auto u2 = Random::Get();

      auto mag         = sigma * sqrt(-2.0 * log(u1));
      out._data[i]     = mag * cos(two_pi * u2) + mu;
      out._data[i + 1] = mag * sin(two_pi * u2) + mu;
    }

    return out;
  }

  /**
   * row getter
   * @returns
   */
  [[nodiscard]] inline size_t rows() const { return _rows; }
  /**
   * columns getter
   * @returns
   */
  [[nodiscard]] inline size_t columns() const { return _columns; }

  /**
   * elements getter
   * @returns
   */
  [[nodiscard]] inline size_t elements() const { return _element_size; }

  /**
   * getter for total number of elements inside matrix
   * @returns number of matrix elements
   */
  [[nodiscard]] inline size_t elements_total() const {return _rows * _columns * _element_size; }

  /**
   * Calculates Determinant
   * @returns
   */
  [[nodiscard]] inline T Determinant() const {
    if(!HasDet()) return 0;

    if(_rows == 3 && _columns == 3) {
      return (
      _data[GetIndex(0, 0)] * _data[GetIndex(1, 1)] * _data[GetIndex(2, 2)]
      + _data[GetIndex(0, 1)] * _data[GetIndex(1, 2)] * _data[GetIndex(2, 0)]
      + _data[GetIndex(0, 2)] * _data[GetIndex(1, 0)] * _data[GetIndex(2, 1)]
      - _data[GetIndex(0, 2)] * _data[GetIndex(1, 1)] * _data[GetIndex(2, 0)]
      - _data[GetIndex(0, 1)] * _data[GetIndex(1, 0)] * _data[GetIndex(2, 2)]
      - _data[GetIndex(0, 0)] * _data[GetIndex(1, 2)] * _data[GetIndex(2, 1)]);
    }
    if(_rows == 2 && _columns == 2) {
      return _data[GetIndex(0, 0)] * _data[GetIndex(1, 1)] - _data[GetIndex(0, 1)] * _data[GetIndex(1, 0)];
    }

    Matrix<T> submat(0.0, _rows - 1, _columns - 1);
    T d = 0;
    {
      for(size_t c = 0; c < _columns; c++) {
        size_t subi = 0; //sub-matrix's i value
        for(size_t i = 1; i < _rows; i++) {
          size_t subj = 0;
          for(size_t j = 0; j < _columns; j++) {
            if(j == c) continue;
            submat._data[submat.GetIndex(subi, subj)] = _data[GetIndex(i, j)];
            subj++;
          }
          subi++;
        }
        d = d + (std::pow(-1, c) * _data[GetIndex(0, c)] * submat.Determinant());
      }
    }
    return d;
  }

  /**
   * Creates transposed matrix of `this`
   * @returns
   */
  [[nodiscard]] constexpr Matrix<T> Transpose() const {
    Matrix<T> res(0, _columns, _rows, _element_size);
    int index_factor = _rows * _element_size;
    for(size_t i = 0; i < (_rows * _columns * _element_size); ++i) {
      res._data[i] = _data[GetIndex(i % (index_factor), i / (index_factor), i % _element_size)];
    }
    return res;
  }

  /**
   * Horizontal matrix concatenation
   * @param other Matrix with same number of rows, dimension n1, m2
   * @returns concatenated matrix of [this, other] with dimension n1, m1 + m2
   */
  Matrix<T> HorizontalConcat(const Matrix<T>& other) {
    assert(this->rows() == other.rows());
    auto result = new Matrix<T>(0, this->rows(), this->columns() + other.columns(), other.elements());
    for(size_t i = 0; i < rows(); ++i) {
      for(size_t j = 0; j < columns() + other.columns(); ++j) {
        for(size_t elem = 0; elem < _element_size; ++elem) {
          (*result)(i, j, elem) = j < columns() ? _data[GetIndex(i, j, elem)] : other(i, j - columns(), elem);
        }
      }
    }
    return *result;
  }

  /** OPERATORS **/

  // Comparison

  /**
   * comparison operator
   * @param rhs
   * @returns
   */
  bool operator==(const Matrix<T>& rhs) const {
    // Just need to check element-wise
    // Dimensions handled by implementation.
    this->assertSize(rhs);
    return elementWiseCompare(rhs);
  }
  /**
   * not-equal operator
   * @param rhs
   * @returns
   */
  bool operator!=(const Matrix<T>& rhs) const {
    return !(rhs == *this); // NOLINT
  }

  bool operator<(const Matrix<T>& rhs) const {
    assertSize(rhs);
    for(size_t i = 0; i < _rows * _columns * _element_size; ++i) {
      if(_data[i] > rhs._data[i]) { return false; }
    }
    return true;
  }

  bool operator>(const Matrix<T>& rhs) const {
    assertSize(rhs);
    for(size_t i = 0; i < _rows * _columns * _element_size; ++i) {
      if(_data[i] < rhs._data[i]) { return false; }
    }
    return true;
  }

  /**
   * Helper to determine whether given matrix is a vector.
   *
   * @returns boolean, true if matrix is vector, else false
   */
  [[nodiscard]] bool IsVector() const { return _columns == 1 || _rows == 1; }

  /**
   * Helper to check for equal dimensions
   * @param other
   */
  void assertSize(const Matrix<T>& other) const {
    assert(_columns == other.columns() && _rows == other.rows() && _element_size == other.elements());
  }

  /**
   * Element-wise comparison
   * @param rhs
   * @returns
   */
  [[nodiscard]] bool elementWiseCompare(const Matrix<T>& rhs) const {
    assertSize(rhs);
    for(size_t i = 0; i < _rows * _columns * _element_size; ++i) {
      if(_data[i] != rhs._data[i]) { return false; }
    }
    return true;
  }


  // Assignment
  /**
   * Assignment-operator.
   *
   * careful! actually overrides different sized matrices, just like other languages (python, matlab)
   * @param other
   * @returns
   */
  Matrix<T> operator=(const Matrix<T>& other) {
    if(this != &other) {
      if((this == NULL) || (_rows != other.rows() || _columns != other.columns())) {
        Resize(other.rows(), other.columns(), other.elements());
      }
      for(size_t i = 0; i < _rows * _columns * _element_size; ++i) { _data[i] = other._data[i]; }
    }
    return *this;
  }

  /**
   * Apply given function to Matrix
   * @param fun element-wise function to apply
   * @returns fun(this)
   */
  Matrix<T> Apply(const std::function<T(T)>& fun) const {
    auto out = (*this);
    for(size_t i = 0; i < _rows * _columns * _element_size; ++i) { out._data[i] = fun(_data[i]); }
    return out;
  }

  // Math

  /**
   * Hadamard Multiplication
   * Z[i][j] = A[i][j] * B[i][j]
   * @param other
   * @returns
   */
  Matrix& HadamardMulti(const Matrix& other) {
    for(size_t i = 0; i < _rows * _columns * _element_size; ++i) { _data[i] *= other._data[i]; }
    return *this;
  }

  /**
   * A form of matrix multiplication
   * For explicit reference please consult https://en.wikipedia.org/wiki/Kronecker_product
   * @param other right hand side with same dimension
   * @returns resulting matrix with same dimension
   */
  Matrix<T>& KroneckerMulti(const Matrix<T>& other) {
    assert(_element_size == other.elements());
    auto result = new Matrix<T>(0, rows() * other.rows(), columns() * other.columns(), _element_size);
    for(size_t m = 0; m < rows(); m++) {
      for(size_t n = 0; n < columns(); n++) {
        for(size_t p = 0; p < other.rows(); p++) {
          for(size_t q = 0; q < other.columns(); q++) {
            for(size_t elem = 0; elem < _element_size; ++elem) {
              (*result)(m * other.rows() + p, n * other.columns() + q, elem) =
              _data[GetIndex(m, n, elem)] * other(p, q, elem);
            }
          }
        }
      }
    }
    return *result;
  }

  /**
   * Calculates sum of all elements
   * @returns element sum
   */
  T sumElements() const {
    T result = T(0.0);
    for(size_t i = 0; i < _rows * _columns * _element_size; ++i) { result += _data[i]; }
    return result;
  }

  /**
   * Calculates element wise sum of sub-elements along given axis
   *
   * @param axis axis index (0: rows, 1: columns) to calculate sum on
   * @returns vector of element wise sums along given axis
   */
  Matrix<T> sum(size_t axis) const {
    Matrix<T> out(0, axis == 0 ? _rows : 1, axis == 1 ? _columns : 1);
    for(size_t i = 0; i < (axis == 0 ? _rows : _columns); ++i) {
      out(axis == 0 ? i : 0, axis == 1 ? i : 0) =
      GetSlice(axis == 0 ? i : 0, axis == 0 ? i : _rows - 1, axis == 1 ? i : 0, axis == 1 ? i : _columns - 1)
      .sumElements();
    }
    return out;
  }

  /**
   * Matrix-Constant-Multiplication
   * @param rhs
   * @returns
   */
  Matrix<T>& operator*=(T rhs) {
    (*this) = *this * rhs;
    return *this;
  }

  /**
   * Matrix-Addition
   * @param rhs
   * @returns
   */
  Matrix<T>& operator+=(const Matrix<T>& rhs) {
    (*this) = (*this) + rhs;
    return *this;
  }
  /**
   * Matrix-Subtraction
   * @param rhs
   * @returns
   */
  Matrix<T>& operator-=(const Matrix<T>& rhs) {
    (*this) = (*this) - rhs;
    return *this;
  }

  // Access

  /**
   * element access
   * @param row row index
   * @param column column index
   * @param elem element index
   * @returns value at given address
   */
  T& operator()(size_t row, size_t column, size_t elem = 0) { return _data[GetIndex(row, column, elem)]; }
  /**
   * const element-access
   * @param row row index
   * @param column column index
   * @param elem element index
   * @returns const value at given address
   */
  T& operator()(size_t row, size_t column, size_t elem = 0) const { return _data[GetIndex(row, column, elem)]; }

  /**
   * row getter
   *
   * **no in-place editing, creates new object!**
   * use SetRow instead
   * @param row index of row to get
   * @returns row elements
   */
  Matrix<T> operator()(size_t row) { return GetSlice(row, row, 0, _columns - 1); }
  /**
   * const row-getter
   * @param row index of row
   * @returns row values
   */
  Matrix<T> operator()(size_t row) const { return GetSlice(row, row, 0, _columns - 1); }

  /**
   * pointer operator
   * @returns
   */
  T& operator*() { return _data[0]; }
  /**
   * const pointer operator
   * @returns
   */
  T& operator*() const { return _data[0]; }

  /**
   * Column setter
   * @param index column index to set
   * @param other new values
   */
  void SetColumn(size_t index, const Matrix<T>& other) {
    bool isInColumns = other.columns() > other.rows();
    auto rowCount    = isInColumns ? other.columns() : other.rows();
    assert(rowCount == _rows);
    assert(_element_size == other.elements());
    for(size_t i = 0; i < rowCount; ++i) {
      for(size_t elem = 0; elem < elements(); ++elem) {
        _data[GetIndex(i, index, elem)] = other(isInColumns ? 0 : i, isInColumns ? i : 0, elem);
      }
    }
  }

  /**
   * Row setter
   * @param index row index to set
   * @param other holds new row elements
   */
  void SetRow(size_t index, const Matrix<T>& other) {
    bool isInColumns = other.columns() > other.rows();
    auto colCount    = isInColumns ? other.columns() : other.rows();
    assert(colCount == _columns);
    assert(_element_size == other.elements());
    for(size_t i = 0; i < colCount; ++i) {
      for(size_t elem = 0; elem < elements(); ++elem) {
        _data[GetIndex(index, i, elem)] = other(isInColumns ? 0 : i, isInColumns ? i : 0, elem);
      }
    }
  }

  /**
   * ostream operator, beatified representation
   * @param ostr
   * @param m
   * @returns
   */
  friend std::ostream& operator<<(std::ostream& ostr, const Matrix& m) {
    ostr.precision(17);
    ostr << "[\n";
    for(size_t row = 0; row < m.rows(); row++) {
      ostr << '\t';
      for(size_t col = 0; col < m.columns(); col++) {
        if(m.elements() > 1) { ostr << "( "; }
        for(size_t elem = 0; elem < m.elements(); elem++) {
          ostr << m._data[m.GetIndex(row, col, elem)];
          if(elem < m.elements() - 1) ostr << ", ";
        }
        if(m.elements() > 1) { ostr << " )"; }
        if(col < m.columns() - 1) ostr << ", ";
      }
      ostr << "\n";
    }
    ostr << "]\n";
    return ostr;
  }

  /**
   * Resizes a matrix
   * @param rows target number of rows
   * @param cols target number of columns
   * @param elementSize target number of elements per cell
   */
  void Resize(size_t rows, size_t cols, size_t elementSize = 1) {
    _rows         = rows;
    _columns      = cols;
    _element_size = elementSize;
    if(_data != nullptr || needsFree) {
      _data = (T*)realloc(_data, rows * cols * elementSize * sizeof(T));
    } else {
      _data = (T*)malloc(rows * cols * elementSize * sizeof(T));
    }
    _dataSize = rows * cols * elementSize;
    needsFree = true;
  }

  /**
   *
   * @param row \f[\in [0, rows() - 1]\f]
   * @param col \f[\in [0, columns() - 1]\f]
   * @param elem \f[\in [0, elements() - 1]\f]
   * @returns elem + col * elements() + row * columns() * elements()
   */
  [[nodiscard]] inline int GetIndex(size_t row, size_t col, size_t elem = 0) const {
    //        assert(row < _rows && col < _columns && elem < _element_size);
    return elem + col * _element_size + row * _columns * _element_size;
  }
  [[nodiscard]] inline Matrix GetSlice(size_t rowStart) const { return GetSlice(rowStart, rowStart, 0, _columns - 1); }
  [[nodiscard]] inline Matrix GetSlice(size_t rowStart, size_t rowEnd) const {
    return GetSlice(rowStart, rowEnd, 0, _columns - 1);
  }
  [[nodiscard]] inline Matrix GetSlice(size_t rowStart, size_t rowEnd, size_t colStart) const {
    return GetSlice(rowStart, rowEnd, colStart, _columns - 1);
  }
  /**
   * Returns a slice of given dimension from the matrix
   *
   * @param rowStart row start index
   * @param rowEnd row end index
   * @param colStart column start index
   * @param colEnd column end index
   * @returns sub-matrix
   */
  [[nodiscard]] inline Matrix GetSlice(size_t rowStart, size_t rowEnd, size_t colStart, size_t colEnd) const {
    size_t numRows = (rowEnd - rowStart) + 1;
    size_t numCols = (colEnd - colStart) + 1;

    Matrix out(0, numRows, numCols, _element_size);

    for(size_t i = 0; i < numRows; ++i) {
      for(size_t j = 0; j < numCols; ++j) {
        for(size_t elem = 0; elem < _element_size; ++elem) {
          out(i, j, elem) = _data[GetIndex(rowStart + i, colStart + j, elem)];
        }
      }
    }
    return out;
  }

  void SetSlice(
  const size_t& row_start,
  const size_t& row_end,
  const size_t& col_start,
  const size_t& col_end,
  const Matrix<T>& slice) {
    size_t numRows = (row_end - row_start) + 1;
    size_t numCols = (col_end - col_start) + 1;
    assert(numRows == slice.rows());
    assert(numCols == slice.columns());

    for(size_t i = 0; i < numRows; ++i) {
      for(size_t j = 0; j < numCols; ++j) { _data[GetIndex(row_start + i, col_start + j)] = slice(i, j); }
    }
  }


  /**
   * Helper method to automatically resolve dimensions through slice
   */
  void SetSlice(const size_t& row_start, const Matrix<T>& slice) {
    return SetSlice(row_start, row_start + slice.rows() - 1, 0, slice.columns() - 1, slice);
  }

  /**
   * returns 1D-Matrix from given index
   * @param index of elements
   * @returns
   */
  Matrix<T> GetComponents(const size_t& index) const {
    assert(index < _element_size);
    Matrix<T> out(0, _rows, _columns, 1);
    for(size_t i = 0; i < _rows; ++i) {
      for(size_t j = 0; j < _columns; ++j) { out(i, j, 0) = _data[GetIndex(i, j, index)]; }
    }
    return out;
  }

  inline Matrix<T> GetSlicesByIndex(const Matrix<size_t>& indices) const {
    assert(indices.IsVector());
    auto _indices = indices.rows() > indices.columns() ? indices : indices.Transpose();
    Matrix<T> out(0, _indices.rows(), _columns, _element_size);
    for(size_t i = 0; i < _indices.rows(); ++i) {
      auto idx   = _indices(i, 0);
      auto slice = GetSlice(idx);
      out.SetSlice(i, slice);
    }
    return out;
  }

private:
  /**
   * Helper to test if Matrix can have a determinant
   * @returns
   */
  [[nodiscard]] bool HasDet() const { return _columns > 1 && _rows > 1 && _element_size == 1; }

  //! number rows
  size_t _rows = 0;
  //! number columns
  size_t _columns = 0;
  //! number elements
  size_t _element_size = 0;

  //! ongoing array representing data
  T* _data = nullptr;
  //! total number of elements
  size_t _dataSize = 0;
  //!
  bool needsFree = false;
};

/**
 * Extra operators
 */
/**
 * Matrix-Addition
 * @tparam T
 * @param lhs
 * @param rhs
 * @returns
 */
template<typename T>
inline Matrix<T> operator+(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  if(rhs.IsVector() && !lhs.IsVector()) {
    bool row_wise = rhs.rows() > rhs.columns();
    if(row_wise) {
      assert(rhs.rows() == lhs.rows());
    } else {
      assert(rhs.columns() == lhs.columns());
    }
    auto result = Matrix<T>(0, lhs.rows(), lhs.columns(), lhs.elements());
    for(size_t i = 0; i < lhs.rows(); i++) {
      for(size_t j = 0; j < lhs.columns(); j++) {
        for(size_t elem = 0; elem < lhs.elements(); elem++) {
          result(i, j, elem) = lhs(i, j, elem) + rhs(row_wise ? i : 0, row_wise ? 0 : j, elem);
        }
      }
    }
    return result;
  }

  lhs.assertSize(rhs);
  auto result = Matrix<T>(0, lhs.rows(), lhs.columns(), lhs.elements());
  for(size_t i = 0; i < lhs.rows(); i++) {
    for(size_t j = 0; j < lhs.columns(); j++) { result(i, j) = lhs(i, j) + rhs(i, j); }
  }
  return result;
}
/**
 * Matrix-Subtraction
 * @tparam T
 * @param lhs
 * @param rhs
 * @returns
 */
template<typename T>
inline Matrix<T> operator-(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  if(rhs.IsVector() && !lhs.IsVector()) {
    // Matrix-Vector substraction
    // substracts vector row/column wise from given lhs matrix
    // similar to numpy.
    auto result = Matrix<T>(0, lhs.rows(), lhs.columns(), lhs.elements());
    for(size_t i = 0; i < lhs.rows(); i++) {
      for(size_t j = 0; j < lhs.columns(); j++) {
        for(size_t elem = 0; elem < lhs.elements(); elem++) {
          result(i, j, elem) =
          lhs(i, j, elem) - rhs(rhs.rows() > rhs.columns() ? i : 0, rhs.rows() > rhs.columns() ? 0 : j, elem);
        }
      }
    }
    return result;
  }
  lhs.assertSize(rhs);
  auto result = Matrix<T>(0, lhs.rows(), lhs.columns(), lhs.elements());
  for(size_t i = 0; i < lhs.rows(); i++) {
    for(size_t j = 0; j < lhs.columns(); j++) {
      for(size_t elem = 0; elem < lhs.elements(); elem++) { result(i, j, elem) = lhs(i, j, elem) - rhs(i, j, elem); }
    }
  }
  return result;
}
/**
 * Scalar Matrix-division
 * @tparam T value type of elements inside given matrix
 * @param lhs scalar divident
 * @param rhs matrix divisor
 * @returns
 */
template<typename T, typename U>
inline Matrix<T> operator/(U lhs, const Matrix<T>& rhs) {
  auto result = Matrix<T>(0.0, rhs.rows(), rhs.columns(), rhs.elements());
  for(size_t i = 0; i < rhs.rows(); i++) {
    for(size_t j = 0; j < rhs.columns(); j++) {
      for(size_t elem = 0; elem < rhs.elements(); elem++) { result(i, j, elem) = lhs / rhs(i, j, elem); }
    }
  }
  return result;
}
/**
 * Element wise division of matrix elements with given scalar
 * @tparam T value type of elements inside given matrix
 * @param lhs matrix divident
 * @param rhs scalar divisor
 * @returns
 */
template<typename T, typename U>
inline Matrix<T> operator/(const Matrix<T>& lhs, const U& rhs) {
  auto result = Matrix<T>(0.0, lhs.rows(), lhs.columns(), lhs.elements());
  for(size_t i = 0; i < lhs.rows(); i++) {
    for(size_t j = 0; j < lhs.columns(); j++) {
      for(size_t elem = 0; elem < lhs.elements(); elem++) { result(i, j, elem) = lhs(i, j, elem) / rhs; }
    }
  }
  return result;
}


/**
 * Matrix-Matrix division, element wise division if rhs is matrix. Row/Column-wise division for given rhs vector.
 *
 * rhs matrix:
 *  lhs: N x M
 *  rhs: M x T
 *  result: N x T
 *
 * rhs vector:
 *  lhs: N x M
 *  rhs: 1 x M or M x 1
 *  result: N x M
 *
 * @tparam value type of matrix elements
 * @param lhs matrix divident
 * @param rhs matrix divisor
 * @retrun matrix in dimension of given
 *
 */
template<typename T>
inline Matrix<T> operator/(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  auto result = Matrix<T>(0.0, lhs.rows(), lhs.columns(), lhs.elements());
  if(rhs.IsVector()) {
    bool row_wise = rhs.rows() > rhs.columns();
    for(size_t i = 0; i < lhs.rows(); i++) {
      for(size_t j = 0; j < lhs.columns(); j++) {
        for(size_t elem = 0; elem < lhs.elements(); elem++) {
          result(i, j, elem) = lhs(i, j, elem) / rhs(row_wise ? i : 0, row_wise ? 0 : j, elem);
        }
      }
    }
    return result;
  }

  return lhs * (1.0 / rhs);
}

/**
 * Simple Matrix scalar multiplication
 * @param lhs
 * @param rhs
 * @returns scaled matrix
 */
template<typename T, typename U>
inline Matrix<T> operator*(const Matrix<T>& lhs, const U& rhs) {
  auto result = Matrix<T>(0.0, lhs.rows(), lhs.columns(), lhs.elements());
  for(size_t i = 0; i < lhs.rows(); i++) {
    for(size_t j = 0; j < lhs.columns(); j++) {
      for(size_t elem = 0; elem < lhs.elements(); elem++) { result(i, j, elem) = lhs(i, j, elem) * rhs; }
    }
  }
  return result;
}
/**
 * Simple Matrix scalar multiplication
 * @param lambda
 * @param A
 * @returns scaled matrix lambda * A = B with B(i, j) = lambda * A(i, j)
 */
template<typename T, typename U>
inline Matrix<T> operator*(U lambda, const Matrix<T>& A) {
  auto result = Matrix<T>(0.0, A.rows(), A.columns(), A.elements());
  for(size_t i = 0; i < A.rows(); i++) {
    for(size_t j = 0; j < A.columns(); j++) {
      for(size_t elem = 0; elem < A.elements(); elem++) { result(i, j, elem) = A(i, j, elem) * lambda; }
    }
  }
  return result;
}

/**
 * Regular Matrix-Matrix multiplication
 * Calculates LHS * RHS
 * @param lhs
 * @param rhs
 * @returns Rows x C result matrix
 */
template<typename T>
inline Matrix<T> operator*(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  if(lhs.columns() == rhs.rows() && lhs.elements() == rhs.elements()) {
    auto result = Matrix<T>(0.0, lhs.rows(), rhs.columns(), rhs.elements());
    for(size_t i = 0; i < lhs.rows(); i++) {
      for(size_t j = 0; j < rhs.columns(); j++) {
        for(size_t k = 0; k < rhs.rows(); k++) {
          for(size_t elem = 0; elem < rhs.elements(); elem++) {
            result(i, j, elem) += (T)(lhs(i, k, elem) * rhs(k, j, elem));
          }
        }
      }
    }
    return result;
  }
  assert(rhs.IsVector() && !lhs.IsVector());

  auto row_wise = rhs.rows() > rhs.columns();
  auto result   = Matrix<T>(0.0, lhs.rows(), lhs.columns(), lhs.elements());
  for(size_t i = 0; i < lhs.rows(); i++) {
    for(size_t j = 0; j < lhs.columns(); j++) {
      for(size_t k = 0; k < lhs.elements(); k++) {
        result(i, j, k) = (T)(lhs(i, j, k) * rhs(row_wise ? i : 0, row_wise ? 0 : j, k));
      }
    }
  }
  return result;
}

/**
 * \example TestMatrix.cpp
 * This is an example on how to use the Matrix class.
 */
