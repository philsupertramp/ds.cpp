#pragma once

#include "Matrix.h"
#include <type_traits>
#include <vector>


/**
 * Insertion sort for numerical vectors of type T
 * @tparam T input value type
 * @param elem element to insert into given vector
 * @param vec given vector to add the element to
 * @param current_index vector index of last element
 */
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
void insertion_sort(const T& elem, std::vector<T>& vec, const int& current_index) {
  int i = current_index;
  for(; i > 1; i--) {
    if(vec[i - 1] < elem) { break; }
    vec[i] = vec[i - 1];
  }
  vec[i] = elem;
}
/**
 * Sort given vector of elements of numerical type T
 *
 * currently using insertion sort, might change at some point
 * @tparam T element type
 * @param in input vector
 * @returns sorted vector with ascending values
 */
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
std::vector<T> sort(const std::vector<T>& in) {
  std::vector<T> out(in.size());
  int n = -1;
  for(auto elem : in) {
    // Insertion sort
    n = n + 1;
    insertion_sort(elem, out, n);
  }
  return out;
}

template<typename T>
void insertion_sort(const T& elem, Matrix<T>& out, const int& current_index) {
  int i = current_index;
  if(current_index > 0) {
    for(; i >= 1; --i) {
      if(out(i - 1, 0) < elem) { break; }
      out(i, 0) = out(i - 1, 0);
    }
  }
  out(i, 0) = elem;
}

template<typename T>
Matrix<T> sort(const Matrix<T>& in) {
  bool requires_transposition = in.rows() < in.columns();
  auto _in                    = requires_transposition ? in.Transpose() : in;
  Matrix<T> out               = Matrix<T>(std::numeric_limits<T>::min(), _in.rows(), 1);
  for(int i = 0; i < _in.rows(); ++i) { insertion_sort(_in(i, 0), out, i); }
  return requires_transposition ? out.Transpose() : out;
}

/**
 * \example TestSorting.cpp
 * This is an example on how to use the sort method.
 */
