#include "../../Test.h"
#include <math/ds/preprocessing/Scaler.h>

class StandardScalerTestCase : public Test
{
  bool TestConstructor() {
    auto clf = StandardScaler();
    return true;
  }

  bool TestFit() {
    Matrix<double> input_data(
    { { 2, 1 }, { 4, 1 }, { 2, 3 }, { 4, 3 }, { 8, 6 }, { 10, 8 }, { 2, 12 }, { 2, 14 }, { 4, 14 } });
    Matrix<double> labels({ { 0 }, { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
    Matrix<double> expected_weights = { { 38.0 / 9.0, 62.0 / 9.0 } };
    Matrix<double> expected_sds     = { { 2.739739556875101, 5.043024763515652 } };
    auto clf                        = StandardScaler();
    clf.fit(input_data, labels);

    AssertEqual(clf.means, expected_weights);
    AssertEqual(clf.std_deviations, expected_sds);

    clf = StandardScaler(false);
    clf.fit(input_data, labels);

    AssertEqual(clf.means, zeros(1, input_data.columns()));
    AssertEqual(clf.std_deviations, expected_sds);

    clf = StandardScaler(true, false);
    clf.fit(input_data, labels);
    AssertEqual(clf.means, expected_weights);
    AssertEqual(clf.std_deviations, ones(1, input_data.columns()));

    clf = StandardScaler(false, false);
    clf.fit(input_data, labels);
    AssertEqual(clf.means, zeros(1, input_data.columns()));
    AssertEqual(clf.std_deviations, ones(1, input_data.columns()));

    return true;
  }

  bool TestTransform() {
    Matrix<double> input_data(
    { { 2, 1 }, { 4, 1 }, { 2, 3 }, { 4, 3 }, { 8, 6 }, { 10, 8 }, { 2, 12 }, { 2, 14 }, { 4, 14 } });
    Matrix<double> labels({ { 0 }, { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
    Matrix<double> test_data({ { 1, 0 }, { 8, 8 }, { 0, 1 }, { 3, 16 } });
    Matrix<double> expected_input({ { -0.81110711, -1.16772952 },
                                    { -0.08111071, -1.16772952 },
                                    { -0.81110711, -0.77114214 },
                                    { -0.08111071, -0.77114214 },
                                    { 1.37888208, -0.17626106 },
                                    { 2.10887847, 0.22032632 },
                                    { -0.81110711, 1.01350109 },
                                    { -0.81110711, 1.41008848 },
                                    { -0.08111071, 1.41008848 } });
    Matrix<double> expected_test_data = { { -1.1761053031980284, -1.3660232126416183 },
                                          { 1.3788820796114813, 0.2203263246196158 },
                                          { -1.541103500742244, -1.1677295204839642 },
                                          { -0.446108908109597, 1.8066758618808498 } };
    auto clf                          = StandardScaler();
    clf.fit(input_data, labels);

    auto transformations = clf.transform(test_data);

    AssertEqual(transformations, expected_test_data);
    AssertEqual(clf.transform(input_data), expected_input);

    return true;
  }


public:
  virtual void run() {
    TestConstructor();
    TestFit();
    TestTransform();
  }
};


class MinMaxScalerTestCase : public Test
{
  bool TestConstructor() {
    auto clf = StandardScaler();
    return true;
  }

  bool TestFit() {
    Matrix<double> input_data(
    { { 2, 1 }, { 4, 1 }, { 2, 3 }, { 4, 3 }, { 8, 6 }, { 10, 8 }, { 2, 12 }, { 2, 14 }, { 4, 14 } });
    Matrix<double> labels({ { 0 }, { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
    auto clf = MinMaxScaler();
    clf.fit(input_data, labels);

    return true;
  }

  bool TestTransform() {
    Matrix<double> input_data(
    { { 2, 1 }, { 4, 1 }, { 2, 3 }, { 4, 3 }, { 8, 6 }, { 10, 8 }, { 2, 12 }, { 2, 14 }, { 4, 14 } });
    Matrix<double> labels({ { 0 }, { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 2 }, { 2 }, { 2 } });
    Matrix<double> test_data({ { 1, 0 }, { 8, 8 }, { 0, 1 }, { 3, 16 } });
    Matrix<double> expected_input({
    { 0., 0. },
    { 0.25, 0. },
    { 0.0, 0.1538461538461539 },
    { 0.25, 0.1538461538461539 },
    { 0.75, 0.3846153846153846 },
    { 1.0, 0.5384615384615385 },
    { 0.0, 0.8461538461538463 },
    { 0.0, 1.0000000000000002 },
    { 0.25, 1.0000000000000002 },
    });
    Matrix<double> expected_test_data = {
      { 0.125, 0.0 },
      { 1.0, 0.5 },
      { 0.0, 0.0625 },
      { 0.375, 1.0 },
    };
    auto clf = MinMaxScaler();
    clf.fit(input_data, labels);

    auto transformations = clf.transform(test_data);

    std::cout << transformations << std::endl;
    AssertEqual(transformations, expected_test_data);
    AssertEqual(clf.transform(input_data), expected_input);

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
  StandardScalerTestCase().run();
  MinMaxScalerTestCase().run();
  return 0;
}
