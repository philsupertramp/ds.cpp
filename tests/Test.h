#pragma once

#include <cassert>
#include <cmath>
#include <memory>
#include <iostream>
#include <math/numerics/utils.h>
#include <math/format.h>
#include <math/Random.h>
#include <math/sorting.h>

#define TESTING_EPS 1e-8
class Test
{
public:
  bool extended      = false;
  virtual void run() = 0;

  template<typename T>
  void AssertEqual(const std::shared_ptr<T>& a, const std::shared_ptr<T>& b){
    assert(a.get() == b.get());
  }

  template<typename T>
  void AssertEqual(std::shared_ptr<T>& a, std::shared_ptr<T>& b) {
    assert(a.get() == b.get());
  }

  template<
  typename U,
  typename T,
  typename = std::enable_if_t<(std::is_arithmetic<T>::value) && (std::is_arithmetic<U>::value)>>
  void AssertEqual(const U& a, const T& b) {
    if((std::isnan(a) && std::isnan(b)) || (std::isinf(a) && std::isinf(b))) { return; }
    assert(std::abs(double(a) - double(b)) <= TESTING_EPS);
  }
  void AssertEqual(const std::string& a, const std::string& b) { assert(a.compare(b) == 0); }
  template<typename T>
  void AssertEqual(const std::vector<T>& a, const std::vector<T>& b) {
    assert(a.size() == b.size());
    for(size_t i = 0; i < a.size(); i++) { assert(fabs(b[i] - a[i]) <= TESTING_EPS); }
  }
  template<typename T>
  void AssertEqual(const Matrix<T>& a, const Matrix<T>& b) {
    a.assertSize(b);
    for(size_t i = 0; i < a.rows(); ++i) {
      for(size_t j = 0; j < a.columns(); ++j) {
        for(size_t c = 0; c < a.elements(); ++c) { AssertEqual(a(i, j, c), b(i, j, c)); }
      }
    }
  }

  void AssertTrue(const bool& a) { assert(a); }
  void AssertFalse(const bool& a) { assert(!a); }

  void AssertLessThenEqual(const double& a, const double& b) { assert(a <= b); }
  void AssertGreaterThenEqual(const double& a, const double& b) { assert(a >= b); }
  void AssertLess(const double& a, const double& b) { assert(a < b); }
  void AssertGreater(const double& a, const double& b) { assert(a > b); }
};
