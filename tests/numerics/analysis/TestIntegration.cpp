#include "../../Test.h"
#include <math/numerics/analysis/Integration.h>


class IntegrationTestCase : public Test
{
  bool TestQuadAdaptive() {
    auto res = quadrature(sin, 0, M_PI, .1, 1.e-50);
    AssertLessThenEqual(fabs(res.first - 2.0), 9.32e-5);
    return true;
  }

public:
  virtual void run() { TestQuadAdaptive(); }
};


int main() {
  IntegrationTestCase().run();
  return 0;
}
