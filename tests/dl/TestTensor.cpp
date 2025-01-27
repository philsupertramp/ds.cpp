#include "../Test.h"
#include <cstring>
#include <math/dl/Tensor.h>
#include <stdexcept>


class TensorTestCase : public Test {
  bool assertEqual(size_t a, size_t b){
    size_t diff = 0;
    if(a > b){
      diff = a - b;
    } else {
      diff = b - a;
    }
    assert(diff <= TESTING_EPS);
    return true;
  }
  bool assertEqual(const std::vector<size_t>& a, const std::vector<size_t>& b){
    assertEqual(a.size(), b.size());
    
    for(size_t i = 0; i < a.size(); ++i){
      assertEqual(a[i], b[i]);
    }
    return true;
  }
  bool assertEqual(const Tensor& a, const Tensor& b){
    // copare shape
    AssertEqual(a.size(), b.size());

    auto shapeA = a.shape();
    auto shapeB = b.shape();
    AssertEqual(shapeA.size(), shapeB.size());

    for(size_t i = 0; i < shapeA.size(); ++i){
      AssertEqual(shapeA[i], shapeB[i]);
    }

    // compare elements
    //
    for(size_t i = 0; i < a.size(); ++i){
      AssertEqual(a._data[i], b._data[i]);
    }
    return true;
  }
  bool TestDefaultConstructor(){
    auto t = Tensor();
    return true;
  }
  bool TestConstructorOnlyShape(){
    std::vector<size_t> shape({1, 16, 16});
    auto t = Tensor(shape);

    assertEqual(t.shape(), shape);
    AssertEqual(t[{0, 0, 0}], 0.0);
    return true;
  }
  bool TestConstructorShapeAndDefaultValue(){
    std::vector<size_t> shape({1, 16, 16});
    auto t = Tensor(shape, 1.2f);

    assertEqual(t.shape(), shape);
    AssertEqual(t[{0, 0, 0}], 1.2f);
    return true;
  }
  bool TestConstructorShapeAndValues(){
    std::vector<size_t> shape({1, 4, 4});
    std::vector<float> values = {
      1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4
    };
    auto t = Tensor(shape, values);

    assertEqual(t.shape(), shape);
    AssertEqual(t[{0, 0, 0}], 1.0);
    AssertEqual(t[{0, 0, 1}], 1.0);
    AssertEqual(t[{0, 1, 0}], 2.0);
    AssertEqual(t[{0, 2, 0}], 3.0);
    AssertEqual(t[{0, 3, 0}], 4.0);
    return true;
  }

  bool TestShape(){
    std::vector<size_t> shape({1, 16, 16});
    auto t = Tensor(shape);

    AssertEqual(t.shape(), shape);
    return true;
  }

  bool TestSize() {
    std::vector<size_t> shape({1, 16, 16});
    auto t = Tensor(shape);

    AssertEqual(t.size(), 16 * 16);
    return true;
  }

  bool TestReshape(){
    std::vector<size_t> shape({1, 16, 16});
    auto t = Tensor(shape);

    std::vector<size_t> newShape({1, 16 * 16});

    t.reshape({1, 16 * 16});
    
    assertEqual(t.shape(), newShape);
    
    try {
      t.reshape({1, 1, 1});
    } catch (std::invalid_argument& ex) {
      AssertTrue(std::strcmp(ex.what(), "New shape doesn't match element count of previous shape.") == 0);
    }
    return true;
  }

  bool TestIndexAccess() {
    auto t = Tensor({2, 2}, {1,2,3,4});

    AssertEqual(t[{0, 0}], 1);
    AssertEqual(t[{0, 1}], 2);
    AssertEqual(t[{1, 0}], 3);
    AssertEqual(t[{1, 1}], 4);
    return true;
  }

  bool TestConstIndexAccess() {
    auto t = Tensor({2, 2,}, {1,2,3,4});

    for(size_t x = 0; x < 2; x++){
      for(size_t y = 0; y < 2; y++){
        const float value = t[{x, y}];
        AssertEqual(value, (float)(x * 2. + y + 1));
      }
    }
    return true;
  }

  bool TestSlice(){
    auto t = Tensor({1, 2, 2}, {1,2,3,4});
    const auto t1 = Tensor({1, 2, 2}, {1,2,3,4});
    const auto t2 = Tensor({2, 2}, {1,2,3,4});

    const auto slice = t.slice({0, 0, 0}, {1, 2, 2});
    assertEqual(slice, t);

    auto newT = t.slice({0,0,0}, {1,2,2});
    newT.reshape({2,2});
    assertEqual(newT, t2);

    const auto t3 = Tensor({1, 1, 2}, {1,2});
    assertEqual(t.slice({0, 0, 0}, {1, 1, 2}), t3);

    // slice start index is equal to end index
    try{
      t.slice({1, 1, 1}, {1, 1, 1});
    } catch (std::out_of_range& ex) {
      AssertTrue(std::strcmp(ex.what(), "Invalid slice range.") == 0);
    }

    try{
      t.slice({1}, {1, 1, 1});
    } catch (std::invalid_argument& ex) {
      AssertTrue(std::strcmp(ex.what(), "Start and end must have the same number of dimensions as the tensor.") == 0);
    }

    return true;
  }

  bool TestAdditionOperator(){
    Tensor A({1}, 5);
    Tensor B({1}, 10);
    Tensor E({1}, 15);

    assertEqual(A + B, E);

    A = Tensor({2, 2}, 1);
    B = Tensor({2, 2}, 5);
    E = Tensor({2, 2}, 6);

    assertEqual(A + B, E);

    A = Tensor({1, 2}, 1);
    B = Tensor({2, 2}, 2);

    bool completed = true;
    try {
      A + B;
    } catch (std::invalid_argument& ex) {
      AssertTrue(std::strcmp(ex.what(), "Wrong shapes.") == 0);
      completed = false;
    }

    AssertFalse(completed);

    return true;
  }

  bool TestInplaceAdditionOperator() {
    Tensor A({1}, 5);
    Tensor B({1}, 10);
    Tensor E({1}, 15);

    A += B;

    assertEqual(A, E);

    A = Tensor({2, 2}, 1);
    B = Tensor({2, 2}, 5);
    E = Tensor({2, 2}, 6);

    A += B;
    assertEqual(A, E);

    A = Tensor({1, 2}, 1);
    B = Tensor({2, 2}, 2);

    bool completed = true;
    try {
      A += B;
    } catch (std::invalid_argument& ex) {
      AssertTrue(std::strcmp(ex.what(), "Wrong shapes.") == 0);
      completed = false;
    }

    AssertFalse(completed);

    return true;
  }

  bool TestSubtractionOperator() {
    Tensor A({1}, 5);
    Tensor B({1}, 10);
    Tensor E({1}, 15);

    assertEqual(E - A, B);

    A = Tensor({2, 2}, 1);
    B = Tensor({2, 2}, 5);
    E = Tensor({2, 2}, 6);

    assertEqual(E - A, B);

    A = Tensor({1, 2}, 1);
    B = Tensor({2, 2}, 2);

    bool completed = true;
    try {
      A - B;
    } catch (std::invalid_argument& ex) {
      completed = false;
      AssertTrue(std::strcmp(ex.what(), "Wrong shapes.") == 0);
    }
    AssertFalse(completed);


    return true;
  }

  bool TestInplaceSubtractionOperator(){
    Tensor A({1}, 5);
    Tensor B({1}, 10);
    Tensor E({1}, 15);

    E -= A;

    assertEqual(A, B);

    A = Tensor({2, 2}, 1);
    B = Tensor({2, 2}, 5);
    E = Tensor({2, 2}, 6);

    E -= A;
    assertEqual(E, B);

    A = Tensor({1, 2}, 1);
    B = Tensor({2, 2}, 2);

    bool completed = true;
    try {
      A -= B;
    } catch (std::invalid_argument& ex) {
      completed = false;
      AssertTrue(std::strcmp(ex.what(), "Wrong shapes.") == 0);
    }
    AssertFalse(completed);

    return true;
  }

  bool TestElementwiseMultiplicationOperator() {
    Tensor A({1, 2}, 1);
    Tensor B({1, 2}, 2);

    Tensor E({1, 2}, 2);

    assertEqual(A * B, E);

    A = Tensor({2, 3, 3}, 1);
    B = Tensor({3, 2, 3}, 2);

    bool completed = true;
    try {
      A * B;
    } catch (std::invalid_argument& ex) {
      AssertTrue(std::strcmp(ex.what(), "Shapes are not broadcastable for multiplication") == 0);
      completed = false;
    }
    AssertFalse(completed);

    return true;
  }

  bool TestElementwiseDivisionOperator(){
    Tensor A({1, 2}, 1);
    Tensor B({1, 2}, 2);

    Tensor E({1, 2}, 2);

    assertEqual(E / A, B);

    A = Tensor({2, 3, 3}, 1);
    B = Tensor({3, 2, 3}, 2);

    bool completed = true;
    try {
      E / B;
    } catch (std::invalid_argument& ex) {
      AssertTrue(std::strcmp(ex.what(), "Shapes are not broadcastable for division") == 0);
      completed = false;
    }
    AssertFalse(completed);

    return true;
  }

  bool TestMatmul(){
    Tensor A({1, 2}, {1, 2});
    Tensor B({2, 1}, {1, 1});

    Tensor E({2, 2}, {1, 2, 1, 2});
    Tensor E2({1, 1}, {3});

    assertEqual(B.matmul(A), E);
    assertEqual(A.matmul(B), E2);

    bool created = true;
    try {
      Tensor({1,1,1}, 1).matmul(Tensor({1,1}));
    } catch (std::invalid_argument& ex) {
      AssertEqual(std::strcmp(ex.what(), "matmul only supports 2D matrices."), 0);
      created = false;
    }
    AssertFalse(created);

    created = true;
    try {
      Tensor({1,10}, 1).matmul(Tensor({1,1}));
    } catch (std::invalid_argument& ex) {
      AssertEqual(std::strcmp(ex.what(), "Incompatible shapes for matrix multiplication."), 0);
      created = false;
    }
    AssertFalse(created);
    return true;
  }

  bool TestTranspose() {
    Tensor A({1, 2}, {1, 2});
    Tensor B({2, 1}, {2, 1});

    assertEqual(A.transpose({1, 0}), B);

    return true;
  }

  bool TestSum() {
    Tensor A({5, 10}, 1);

    std::cout << A << std::endl;
    std::cout << A.sum(0) << std::endl;
    std::cout << A.sum(-1) << std::endl;

    Tensor E({5,}, 10);
    Tensor E2({10,}, 5);

    assertEqual(A.sum(0), E);
    assertEqual(A.sum(-1), E2);

    AssertFalse(true);
    //AssertEqual(A.sum(0), 50);
    //AssertEqual(A.sum(1), 50);

    return true;
  }

public:
  void run() {
    TestDefaultConstructor();
    TestConstructorOnlyShape();
    TestConstructorShapeAndDefaultValue();
    TestConstructorShapeAndValues();
    TestShape();
    TestReshape();
    TestSize();
    TestIndexAccess();
    TestConstIndexAccess();
    TestSlice();

    TestAdditionOperator();
    TestInplaceAdditionOperator();
    TestSubtractionOperator();
    TestInplaceSubtractionOperator();
    TestElementwiseMultiplicationOperator();
    TestElementwiseDivisionOperator();

    TestMatmul();
    TestTranspose();

    TestSum();
  }
};

int main() {
  TensorTestCase().run();
  return 0;
}

