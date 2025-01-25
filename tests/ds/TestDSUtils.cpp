#include "../Test.h"
#include <math/ds/utils.h>


class DSUtilsTestCase : public Test
{
  bool TestUtils() {
    AssertEqual(relu(-1), 0);
    AssertEqual(relu(1), 1);
    return true;
  }

public:
  void run() override { TestUtils(); }
};

int main() { DSUtilsTestCase().run(); }