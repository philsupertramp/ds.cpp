#pragma once

#include "Matrix.h"

/**
 * Helper utilities
 */

/**
 * Element wise multiplication
 * @tparam T value type
 * @param lhs left hand side with dimension n1, m1
 * @param rhs right hand side with dimension n1, m1
 * @returns product of element wise multiplication of lhs * rhs with dimension n1, m1
 */
template<typename T>
Matrix<T> HadamardMulti(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  lhs.assertSize(rhs);
  auto result = Matrix<T>(0, lhs.rows(), lhs.columns(), lhs.elements());
  for(size_t i = 0; i < result.rows(); i++) {
    for(size_t j = 0; j < result.columns(); j++) {
      for(size_t elem = 0; elem < result.elements(); elem++) { result(i, j, elem) = lhs(i, j, elem) * rhs(i, j, elem); }
    }
  }
  return result;
}

/**
 * Element wise division
 * @tparam T value type
 * @param lhs left hand side with dimension n1, m1
 * @param rhs right hand side with dimension n1, m1
 * @returns product of element wise multiplication of lhs * rhs with dimension n1, m1
 */
template<typename T>
Matrix<T> HadamardDiv(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  lhs.assertSize(rhs);
  auto result = Matrix<T>(0, lhs.rows(), lhs.columns(), lhs.elements());
  for(size_t i = 0; i < result.rows(); i++) {
    for(size_t j = 0; j < result.columns(); j++) {
      for(size_t elem = 0; elem < result.elements(); elem++) { result(i, j, elem) = lhs(i, j, elem) / rhs(i, j, elem); }
    }
  }
  return result;
}

/**
 * A form of matrix multiplication
 * For explicit reference please consult https://en.wikipedia.org/wiki/Kronecker_product
 * @tparam T value type
 * @param lhs left hand side with dimension n1, m1
 * @param rhs right hand side with dimension n2, m2
 * @returns resulting matrix with dimension n1 * n2, m1 * m2
 */
template<typename T>
Matrix<T> KroneckerMulti(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  assert(lhs.elements() == rhs.elements());
  auto result = Matrix<T>(0, lhs.rows() * rhs.rows(), lhs.columns() * rhs.columns(), rhs.elements());
  for(size_t m = 0; m < lhs.rows(); m++) {
    for(size_t n = 0; n < lhs.columns(); n++) {
      for(size_t p = 0; p < rhs.rows(); p++) {
        for(size_t q = 0; q < rhs.columns(); q++) {
          for(size_t elem = 0; elem < rhs.elements(); elem++) {
            result(m * rhs.rows() + p, n * rhs.columns() + q, elem) = lhs(m, n, elem) * rhs(p, q, elem);
          }
        }
      }
    }
  }
  return result;
}

/**
 * Horizontal concatenation of 2 matrices of same Row size
 * @tparam T value type
 * @param lhs left hand side with dimension n1, m1
 * @param rhs right hand side with dimension n1, m2
 * @returns concatenated matrix [lhs, rhs] of dimension n1, m1 + m2
 */
template<typename T>
Matrix<T> HorizontalConcat(const Matrix<T>& lhs, const Matrix<T>& rhs) {
  assert(lhs.rows() == rhs.rows());
  assert(lhs.elements() == rhs.elements());
  auto result = Matrix<T>(0.0, lhs.rows(), lhs.columns() + rhs.columns(), lhs.elements());
  for(size_t i = 0; i < lhs.rows(); ++i) {
    for(size_t j = 0; j < lhs.columns() + rhs.columns(); ++j) {
      for(size_t elem = 0; elem < lhs.elements(); ++elem) {
        result(i, j, elem) = j < lhs.columns() ? lhs(i, j, elem) : rhs(i, j - lhs.columns(), elem);
      }
    }
  }
  return result;
}

/**
 * Counts correlations between `A` and `B`
 * @tparam T value type of `A` and `B`
 * @param A matrix to compare
 * @param B matrix to compare with
 * @returns number of element wise equal elements
 */
template<typename T>
size_t Corr(const Matrix<T>& A, const Matrix<T>& B) {
  A.assertSize(B);
  size_t count = 0;
  for(size_t i = 0; i < A.rows(); i++) {
    for(size_t j = 0; j < A.columns(); j++) {
      for(size_t elem = 0; elem < A.elements(); elem++) { count += (A(i, j, elem) == B(i, j, elem)); }
    }
  }
  return count;
}

/**
 * Converts array of elements of type `T` into matrix of given `size`
 * @tparam T value type
 * @param value pointer to array of elements of type `T`
 * @param size dimension of target matrix
 * @returns
 */
template<typename T>
Matrix<T> from_vptr(const T* value, MatrixDimension size) {
  auto out = Matrix<T>(0, size.rows, size.columns);
  for(size_t i = 0; i < size.rows; i++) {
    for(size_t j = 0; j < size.columns; j++) {
      for(size_t elem = 0; elem < size.elemDim; elem++) {
        out(i, j, elem) = value[elem + j * size.elemDim + i * size.columns * size.elemDim];
      }
    }
  }
  return out;
}

/**
 * Search index of value with maximum value
 * **Caution!** This value does represent the index in a ongoing array.
 *
 * @tparam T
 * @param mat
 * @returns
 */
template<typename T>
size_t argmax(const Matrix<T>& mat) {
  T maxVal        = std::numeric_limits<T>::min();
  size_t maxIndex = -1;
  for(size_t i = 0; i < mat.rows(); i++) {
    for(size_t j = 0; j < mat.columns(); j++) {
      for(size_t elem = 0; elem < mat.elements(); elem++) {
        if(mat(i, j, elem) > maxVal) {
          maxVal   = mat(i, j, elem);
          maxIndex = elem + j * mat.elements() + i * mat.columns() * mat.elements();
        }
      }
    }
  }
  return maxIndex;
}

/**
 * Search index of value with lowest value
 * **Caution!** This value does represent the index in a ongoing array.
 * @tparam T matrix value type
 * @param mat element to search in
 * @returns index of minimal value
 */
template<typename T>
size_t argmin(const Matrix<T>& mat) {
  T maxVal        = std::numeric_limits<T>::max();
  size_t maxIndex = -1;
  for(size_t i = 0; i < mat.rows(); i++) {
    for(size_t j = 0; j < mat.columns(); j++) {
      for(size_t elem = 0; elem < mat.elements(); ++elem) {
        if(mat(i, j, elem) < maxVal) {
          maxVal   = mat(i, j, elem);
          maxIndex = elem + j * mat.elements() + i * mat.columns() * mat.elements();
        }
      }
    }
  }
  return maxIndex;
}


/**
 * Returns elements chosen from `valIfTrue` or `valIfFalse` depending on `condition`
 * @tparam T value type
 * @param condition lambda/function to test a condition on every element of `in`
 * @param in input values
 * @param valIfTrue value vector with values in case `in` meets `condition`
 * @param valIfFalse value vector with values in case `in` does not meet `condition`
 * @returns
 */
template<typename T>
Matrix<T> where(
const std::function<bool(T)>& condition, const Matrix<T>& in, const Matrix<T>& valIfTrue, const Matrix<T>& valIfFalse) {
  assert(valIfTrue.columns() == valIfFalse.columns() && valIfTrue.rows() == valIfFalse.rows());
  bool refVector = true;
  if((valIfTrue.columns() == valIfTrue.rows()) == 1) { refVector = false; }
  auto out = refVector ? valIfTrue : Matrix<T>(0, in.rows(), in.columns(), in.elements());

  for(size_t i = 0; i < in.rows(); i++) {
    for(size_t j = 0; j < in.columns(); j++) {
      for(size_t elem = 0; elem < in.elements(); elem++) {
        if(refVector) {
          if(!condition(in(i, j, elem))) out(i, j, elem) = valIfFalse(i, j);
          else
            out(i, j, elem) = in(i, j, elem);
        } else {
          out(i, j, elem) = condition(in(i, j, elem)) ? valIfTrue(0, 0) : valIfFalse(0, 0);
        }
      }
    }
  }
  return out;
}

/**
 * Evaluates elements of a given vector. Responds with indices of
 * true values.
 *
 * @param in vector to evaluate in
 * @return vector of indices where element has true value
 */
template<typename T>
Matrix<size_t> where_value(const Matrix<T>& in, T value) {
  assert(in.IsVector());
  bool requires_transposition = in.rows() < in.columns();
  Matrix<size_t> out          = Matrix<size_t>(0, !requires_transposition ? in.rows() : in.columns(), 1);
  size_t found_vals           = 0;
  for(size_t i = 0; i < (!requires_transposition ? in.rows() : in.columns()); ++i) {
    if(in(requires_transposition ? 0 : i, requires_transposition ? i : 0) == value) {
      out(found_vals, 0) = i;
      found_vals++;
    }
  }
  return requires_transposition ? out.GetSlice(0, found_vals - 1).Transpose() : out.GetSlice(0, found_vals - 1);
}
/**
 * Evaluates elements of a given vector. Responds with indices of
 * true values.
 *
 * @param in vector to evaluate in
 * @return vector of indices where element has true value
 */
template<typename T>
Matrix<size_t> where_true(const Matrix<T>& in) {
  return where_value(in, 1.0);
}
/**
 * Evaluates elements of a given vector. Responds with indices of
 * true values.
 *
 * @param in vector to evaluate in
 * @return vector of indices where element has true value
 */
template<typename T>
Matrix<size_t> where_false(const Matrix<T>& in) {
  return where_value(in, 0.0);
}


/**
 * Converts two input matrices into a vector of
 * row-wise pairs of `a` and `b`
 * @tparam T value type
 * @param a
 * @param b
 * @returns
 */
template<typename T>
std::vector<std::pair<Matrix<T>, Matrix<T>>> zip(const Matrix<T>& a, const Matrix<T>& b) {
  std::vector<std::pair<Matrix<T>, Matrix<T>>> out(a.rows());
  for(size_t i = 0; i < a.rows(); i++) {
    Matrix<T> subA, subB;
    subA.Resize(1, a.columns());
    subB.Resize(1, b.columns());
    for(size_t j = 0; j < a.columns(); j++) { subA(0, j) = a(i, j); }
    for(size_t j = 0; j < b.columns(); j++) { subB(0, j) = b(i, j); }

    out[i] = { subA, subB };
  }
  return out;
}

/**
 * Max value of given Matrix
 * @tparam T given datatype
 * @param mat matrix to search max value in
 * @returns max value of given matrix
 */
template<typename T>
T max(const Matrix<T>& mat) {
  T maxVal = std::numeric_limits<T>::min();
  for(size_t i = 0; i < mat.rows(); i++) {
    for(size_t j = 0; j < mat.columns(); j++) {
      for(size_t k = 0; k < mat.elements(); k++) {
        if(mat(i, j, k) > maxVal) { maxVal = mat(i, j, k); }
      }
    }
  }
  return maxVal;
}


template<typename T>
Matrix<T> max(const Matrix<T>& mat, int axis) {
  bool row_wise = axis == 0;
  Matrix<T> out = Matrix<T>(0, row_wise ? 1 : mat.rows(), row_wise ? mat.columns() : 1);

  for(size_t i = 0; i < (row_wise ? mat.columns() : mat.rows()); i++) {
    out(row_wise ? 0 : i, row_wise ? i : 0) = elemMax(
    mat.GetSlice(row_wise ? 0 : i, row_wise ? mat.rows() - 1 : i, row_wise ? i : 0, row_wise ? i : mat.columns() - 1),
    0);
  }
  return out;
}
/**
 * Min value of given matrix
 * @tparam T given datatype
 * @param mat matrix to search min value in
 * @returns min value of matrix
 */
template<typename T>
T min(const Matrix<T>& mat) {
  T minVal = std::numeric_limits<T>::max();
  for(size_t i = 0; i < mat.rows(); i++) {
    for(size_t j = 0; j < mat.columns(); j++) {
      for(size_t k = 0; k < mat.elements(); k++) {
        if(mat(i, j, k) < minVal) { minVal = mat(i, j, k); }
      }
    }
  }
  return minVal;
}

template<typename T>
Matrix<T> min(const Matrix<T>& mat, int axis) {
  bool row_wise = axis == 0;
  Matrix<T> out = Matrix<T>(0, row_wise ? 1 : mat.rows(), row_wise ? mat.columns() : 1);

  for(size_t i = 0; i < (row_wise ? mat.columns() : mat.rows()); i++) {
    out(row_wise ? 0 : i, row_wise ? i : 0) = elemMin(
    mat.GetSlice(row_wise ? 0 : i, row_wise ? mat.rows() - 1 : i, row_wise ? i : 0, row_wise ? i : mat.columns() - 1),
    0);
  }
  return out;
}


/**
 * Max value from given element index in matrix
 * @tparam T given datatype
 * @param mat matrix to search in
 * @param elemIndex index of element to compute max value of
 * @returns max value over all elements with given index
 */
template<typename T>
T elemMax(const Matrix<T>& mat, const size_t& elemIndex) {
  assert(mat.elements() - 1 >= elemIndex);
  T maxVal     = std::numeric_limits<T>::min();
  size_t index = 0;
  for(size_t i = 0; i < mat.rows(); i++) {
    for(size_t j = 0; j < mat.columns(); j++) {
      if(mat(i, j, elemIndex) > maxVal) { maxVal = mat(i, j, elemIndex); }
      index++;
    }
  }
  return maxVal;
}

/**
 * Min value from given element index in matrix
 * @tparam T given datatype
 * @param mat matrix to search in
 * @param elemIndex index of element to compute max value of
 * @returns min value over all elements with given index
 */
template<typename T>
T elemMin(const Matrix<T>& mat, const size_t& elemIndex) {
  assert(mat.elements() - 1 >= elemIndex);
  T maxVal     = std::numeric_limits<T>::max();
  size_t index = 0;
  for(size_t i = 0; i < mat.rows(); i++) {
    for(size_t j = 0; j < mat.columns(); j++) {
      if(mat(i, j, elemIndex) < maxVal) { maxVal = mat(i, j, elemIndex); }
      index++;
    }
  }
  return maxVal;
}

/**
 *  Calculates mean over given axis
 *
 * @tparam T given dataatype
 * @param mat matrix to calculate mean of
 * @param axis axis along which to calculate the mean (-1: no axis - over all elements, 0: row wise, 1: column wise)
 * @returns mean of elements inside given matrix along given axis
 */
template<typename T>
Matrix<T> mean(const Matrix<T>& mat, int axis = -1) {
  if(axis == -1) {
    Matrix<T> sum = Matrix<T>(0, 1, 1);
    T index       = 0;
    for(size_t i = 0; i < mat.rows(); i++) {
      for(size_t j = 0; j < mat.columns(); j++) {
        sum(0, 0) += mat(i, j);
        index++;
      }
    }
    return (1.0 / index) * sum;
  }
  bool row_wise = axis == 0;

  Matrix<T> sum = Matrix<T>(0, row_wise ? 1 : mat.rows(), row_wise ? mat.columns() : 1);
  for(size_t i = 0; i < (row_wise ? mat.rows() : mat.columns()); i++) {
    sum +=
    mat.GetSlice(row_wise ? i : 0, row_wise ? i : mat.rows() - 1, row_wise ? 0 : i, row_wise ? mat.columns() - 1 : i);
  }
  return (1.0 / (row_wise ? mat.rows() : mat.columns())) * sum;
}

/**
 * mean operation on element with given index
 * @tparam T given datatype
 * @param mat input matrix
 * @param elemIndex index of element to compute mean of
 * @returns mean of all elements
 */
template<typename T>
T elemMean(const Matrix<T>& mat, const size_t& elemIndex) {
  assert(mat.elements() - 1 >= elemIndex);
  T sum(0);
  size_t index = 0;
  for(size_t i = 0; i < mat.rows(); i++) {
    for(size_t j = 0; j < mat.columns(); j++) {
      sum += mat(i, j, elemIndex);
      index++;
    }
  }
  return sum / index;
}

/**
 * Returns Vector of diagonal elements from matrix.
 * @tparam T given datatype
 * @param in input matrix
 * @returns vector of diagonal elements
 */
template<typename T>
Matrix<T> diag_elements(const Matrix<T>& in) {
  Matrix<T> out(0, in.rows(), 1, in.elements());
  for(size_t i = 0; i < in.rows(); i++) {
    for(size_t elem = 0; elem < in.elements(); elem++) { out(i, 0, elem) = in(i, i, elem); }
  }
  return out;
}

/**
 *  Returns unique values of given matrix.
 *  @tparam T given datatype
 *  @param in input matrix
 *  @returns matrix of unique values
 */
template<typename T>
Matrix<T> unique(const Matrix<T>& in, [[maybe_unused]] int axis = 0) {
  bool row_wise     = axis == 0;
  Matrix<T> out     = Matrix<T>(0, in.rows(), in.columns());
  size_t found_vals = 0;
  for(size_t i = 0; i < (row_wise ? in.rows() : in.columns()); ++i) {
    bool found = false;
    auto xi =
    in.GetSlice(row_wise ? i : 0, row_wise ? i : in.rows() - 1, row_wise ? 0 : i, row_wise ? in.columns() - 1 : i);
    for(size_t j = 0; j < found_vals; ++j) {
      found = xi
              == out.GetSlice(
              row_wise ? j : 0, row_wise ? j : out.rows() - 1, row_wise ? 0 : j, row_wise ? out.columns() - 1 : j);
      if(found) { break; }
    }
    if(!found) {
      out.SetSlice(found_vals, xi);
      found_vals++;
    }
  }
  return out.GetSlice(0, found_vals - 1);
}
