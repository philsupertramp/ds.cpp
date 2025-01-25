#include "../../Test.h"
#include <math/ds/decomposition/PCA.h>


class PCATestCase : public Test
{
  bool TestConstructor() {
    auto clf = PCA();
    AssertEqual(clf.keep_components, 0);

    clf = PCA(1);

    AssertEqual(clf.keep_components, 1);
    return true;
  }

  bool TestFit() {
    auto clf = PCA();
    return true;
  }

  bool TestTransform() {
    auto clf = PCA();
    return true;
  }

public:
  virtual void run() {
    TestConstructor();
    TestFit();
    TestTransform();
  }
};

int main() {
  PCATestCase().run();
  return 0;
}
