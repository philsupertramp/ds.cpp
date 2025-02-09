#include "../Test.h"
#include <math/dl/tensor.h>


class TensorTestCase : public Test {
  bool AssertEqual(size_t a, int b){
    assert(a == (size_t)b);
    return true;
  }
  bool AssertEqual(const Tensor& a, const Tensor& b){
    assert(&a == &b);
    return true;
  }
  bool AssertEqual(Tensor* a, Tensor* b){
    assert(a == b);
    return true;
  }
public:

  bool TestConstructors() {
    size_t num_elements[2] = {
      2, 2
    };
    float data[4] = {
      1., 2., 3., 4.
    };
    Tensor T(2, num_elements, (void*)data, datatypes::TYPE_FLOAT);
    num_elements[0] = 1;
    float smallData[2] = {
      1.0, 1.0
    };
    Tensor A(2, num_elements, (void*)smallData, datatypes::TYPE_FLOAT);


    AssertEqual(T.nrows(), 2);
    AssertEqual(A.nrows(), 2);
    return true;
  }

  bool TestAddition(){
    size_t num_elements[2] = {
      2, 2
    };
    float data[4] = {
      1., 2., 3., 4.
    };
    Tensor T(2, num_elements, (void*)data, datatypes::TYPE_FLOAT);
    Tensor T2(2, num_elements, (void*)data, datatypes::TYPE_FLOAT);

    auto newT = add(&T, &T2);
    AssertEqual(*newT->srcL(), T);
    AssertEqual(*newT->srcR(), T2);


    const struct ComputeParams params = {

    };

    return true;
  }

  void run(){
    TestConstructors();
    TestAddition();
  }
};

int main() {
  TensorTestCase().run();
  return 0;
}

