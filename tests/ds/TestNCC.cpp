#include "../Test.h"
#include <math/ds/NCC.h>

class NCCTestCase : public Test
{
  bool TestConstructor() {
    auto clf = NCC(true);
    return true;
  }

  bool TestFit() {
    Matrix<double> input_data(
    { { 2, 1 }, { 4, 1 }, { 2, 3 }, { 4, 3 }, { 8, 6 }, { 10, 8 }, { 2, 12 }, { 2, 14 }, { 4, 14 } });
    Matrix<double> labels({ { 0 }, { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
    Matrix<double> expected_weights({ { 3, 2 }, { 9, 7 }, { 8. / 3, 40. / 3 } });

    auto clf = NCC(true);
    clf.fit(input_data, labels);

    AssertEqual(clf.weights, expected_weights);

    clf = NCC(false);
    clf.fit(input_data, labels);

    AssertEqual(clf.weights, expected_weights);

    return true;
  }

  bool TestPredict() {
    Matrix<double> input_data(
    { { 2, 1 }, { 4, 1 }, { 2, 3 }, { 4, 3 }, { 8, 6 }, { 10, 8 }, { 2, 12 }, { 2, 14 }, { 4, 14 } });
    Matrix<double> labels({ { 0 }, { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
    Matrix<double> test_data({ { 1, 0 }, { 8, 8 }, { 0, 1 }, { 3, 16 } });
    Matrix<double> expected_labels({ { 0 }, { 1 }, { 0 }, { 2 } });

    auto clf = NCC(true);
    clf.fit(input_data, labels);

    AssertEqual(clf.predict(test_data), expected_labels);

    return true;
  }


public:
  virtual void run() {
    TestConstructor();
    TestFit();
    TestPredict();
  }
};

int main() {
  NCCTestCase().run();
  return 0;
}
