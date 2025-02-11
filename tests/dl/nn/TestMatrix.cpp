#include "../../Test.h"
#include <math/dl/nn/matrix.h>
#include <math/dl/nn/engine.h>


class MatrixTestCase : public Test {
  bool TestOperator(){
    // Example: Multiply two 128x128 matrices.
    const size_t dim = 128;
    matrix<double> A(dim, dim, 1.0);
    matrix<double> B(dim, dim, 2.0);

    // Initialize A with values: A(i,j) = i + 1.
    for (size_t i = 0; i < A.numRows(); ++i)
        for (size_t j = 0; j < A.numCols(); ++j)
            A(i, j) = static_cast<double>(i + 1);
    
    // Initialize B with values: B(i,j) = j + 1.
    for (size_t i = 0; i < B.numRows(); ++i)
        for (size_t j = 0; j < B.numCols(); ++j)
            B(i, j) = static_cast<double>(j + 1);
    
    matrix<double> C = A * B;

    std::cout << "matrix C (result of A * B) - first 5 rows and columns:\n";
    for (size_t i = 0; i < std::min(C.numRows(), size_t(5)); ++i) {
        for (size_t j = 0; j < std::min(C.numCols(), size_t(5)); ++j)
            std::cout << C(i, j) << " ";
        std::cout << "\n";
    }

    return true;
  }

  bool TestWithValue(){
    matrix<float> data(2, 2, 1.0f);
    //auto val = std::make_shared<Value<matrix<float>>>(data);

    return true;
  }

public:
  void run(){
    TestOperator();
    TestWithValue();
  }
};

int main() {
  MatrixTestCase().run();
  return 0;
}



