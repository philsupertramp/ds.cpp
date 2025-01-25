#include "Test.h"
#include <math/Matrix.h>
#include <math/matrix_utils.h>


class MatrixTestCase : public Test
{
  bool TestMatrixMultiplication() {
    Matrix<double> A(1.0, 3, 2);
    Matrix<double> B(2.0, 2, 3);
    Matrix<double> C(2.0, 2, 2);
    Matrix<double> C2(1.0, 2, 2);
    /**
     * 1 1    2 2 2
     * 1 1    2 2 2
     * 1 1
     */
    double v = 2.0;
    auto D   = A * v;
    D.assertSize(A);
    for(size_t i = 0; i < D.rows(); ++i) {
      for(size_t j = 0; j < D.columns(); ++j) { assert(D(i, j) == 2.0); }
    }
    D = A;
    D *= v;
    D.assertSize(A);
    for(size_t i = 0; i < D.rows(); ++i) {
      for(size_t j = 0; j < D.columns(); ++j) { assert(D(i, j) == 2.0); }
    }

    //    B * C; <- not possible
    C* C2;

    Matrix<double> C3;
    C3 = C * C2;
    C3.assertSize(C);
    for(size_t i = 0; i < C3.rows(); ++i) {
      for(size_t j = 0; j < C3.columns(); ++j) { assert(C3(i, j) == 4.0); }
    }
    auto E = A * B;
    //        std::cout << E;
    assert(E.columns() == 3);
    assert(E.rows() == 3);
    for(size_t i = 0; i < E.rows(); ++i) {
      for(size_t j = 0; j < E.columns(); ++j) { assert(E(i, j) == 4); }
    }

    Matrix<double> vec(1.0, 10, 1);
    Matrix<double> vec2(2.0, 1, 10);

    auto resVec = vec * vec2;
    assert(resVec.rows() == 10);
    assert(resVec.columns() == 10);
    auto resVec2 = vec2 * vec;
    assert(resVec2.rows() == 1);
    assert(resVec2.columns() == 1);

    Matrix<double> vec2A(2.0, 2, 1);
    Matrix<double> vec2B(2.0, 1, 2);
    Matrix<double> mat2(1.0, 2, 2);
    Matrix<double> mat2B(2.0, 2, 2);
    Matrix<double> vec2b(4.0, 2, 1);
    Matrix<double> resA = mat2 * vec2A;
    Matrix<double> resB = mat2 * vec2B;

    assert(resA == vec2b);
    assert(resB == mat2B);
    return true;
  }

  bool TestMatrixDivision() {
    Matrix<double> A({ { 2, 4, 8 } });
    Matrix<double> B({ { .5, .25, .125 } });
    Matrix<double> C({ { 1, 2, 4 } });
    AssertEqual(1.0 / A, B);
    AssertEqual(A / 2, C);
    AssertEqual(1 / A, B);
    AssertEqual(A / 2.0f, C);

    auto E = Matrix<double>(2, 2, 2);
    auto F = Matrix<double>(1, 2, 2);
    AssertEqual(E / F, 2 * E);

    return true;
  }

  bool TestMatrixAddition() {
    Matrix<double> A(2.0, 2, 2);
    Matrix<double> B(2.0, 2, 2);
    Matrix<double> C(0, 2, 2);
    auto D = A + B;

    Matrix<double> E(4.0, 2, 2);
    assert(D == E);
    C += A;
    assert(C == B);

    Matrix<double> vec2(1.0, 2, 1);
    Matrix<double> vec2b(1.0, 2, 1);
    Matrix<double> vec2c(2.0, 2, 1);

    assert((vec2 + vec2b) == vec2c);

    // Impossible
    //    vec2 + A;
    auto foo = A + vec2;
    Matrix<double> F(3.0, 2, 2);
    AssertEqual(foo, F);
    AssertEqual(A + vec2c, E);
    return true;
  }

  bool TestHadamardMultiplication() {
    Matrix<double> A(1.0, 2, 2);
    Matrix<double> B(2.015, 2, 2);

    auto C = A.HadamardMulti(B);

    assert(C == B);

    Matrix<double> D({ { 0, 1 }, { 0, 0 } });
    Matrix<double> D2({ { 0, 1 }, { 0, 0 } });
    Matrix<double> E(5.0, 2, 2);
    Matrix<double> resultA({ { 0, 5.0 }, { 0, 0 } });
    Matrix<double> resultB({ { 0, 25.0 }, { 0, 0 } });

    assert(D.HadamardMulti(E) == resultA);
    D.HadamardMulti(E);
    assert(D == resultB);

    auto resD2E = HadamardMulti(D2, E);
    assert(resD2E == resultA);

    Matrix<double> Q1(10.0, 2, 5);
    Matrix<double> Q2(10.0, 3, 5);
    Matrix<double> Q3(10.0, 5, 2);

    // Impossible
    //    Q1.HadamardMulti(Q2);
    //    Q1.HadamardMulti(Q3);
    //    HadamardMulti(Q1, Q2);
    return true;
  }

  bool TestKroneckerMultiplication() {
    Matrix<double> A(1.0, 1, 2);
    Matrix<double> B(2.0, 2, 1);
    Matrix<double> C(2.0, 2, 2);

    auto resA = A.KroneckerMulti(B);

    assert(resA == C);

    Matrix<double> A2({ { 1, 2 }, { 3, 4 } });
    Matrix<double> B2({ { 0, 5 }, { 6, 7 } });
    Matrix<double> C2({ { 0, 5, 0, 10 }, { 6, 7, 12, 14 }, { 0, 15, 0, 20 }, { 18, 21, 24, 28 } });

    Matrix<double> resB = A2.KroneckerMulti(B2);

    assert(resB == C2);

    Matrix<double> A3({ { 1, 2 }, { 3, 4 }, { 5, 6 } });
    Matrix<double> A32({ { 1, 2 }, { 3, 4 }, { 5, 6 } });
    Matrix<double> B3({ { 7, 8 }, { 9, 0 } });
    Matrix<double> C3(
    { { 7, 8, 14, 16 }, { 9, 0, 18, 0 }, { 21, 24, 28, 32 }, { 27, 0, 36, 0 }, { 35, 40, 42, 48 }, { 45, 0, 54, 0 } });

    auto resC = A3.KroneckerMulti(B3);
    assert(resC == C3);

    auto resC2 = KroneckerMulti(A32, B3);
    assert(resC2 == resC && resC2 == C3);

    return true;
  }

  bool TestHorizontalConcat() {
    Matrix<double> A(1.0, 2, 2);
    Matrix<double> A2(1.0, 2, 2);
    Matrix<double> B(2.0, 2, 1);
    Matrix<double> C({ { 1.0, 1.0, 2.0 }, { 1.0, 1.0, 2.0 } });

    auto res1 = A.HorizontalConcat(B);

    assert(res1 == C);

    auto res2 = HorizontalConcat(A2, B);

    assert(res2 == C);

    return true;
  }

  bool TestMatrixCompare() {
    Matrix<double> A(2.0, 2, 2);
    Matrix<double> B(2.0, 2, 2);
    Matrix<double> C(0, 2, 2);
    Matrix<double> D(2.0, 2, 2);
    Matrix<double> E(2.0, 2, 2);
    Matrix<double> F({ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } });
    Matrix<double> G({ { 1, 0, 0 }, { 0, 1, 1 }, { 0, 0, 1 } });

    assert(A == B);
    assert(A != C);
    // cannot be compared by implementation
    //    assert(A != D);
    //    assert(A != E);
    //    assert(D == E);
    assert(F != G);
    assert(F.Determinant() == G.Determinant());

    AssertTrue(A < B);
    AssertTrue(B < A);
    AssertTrue(A > B);
    AssertTrue(B > A);
    AssertTrue(A > C);
    AssertFalse(A < C);
    AssertTrue(C < A);
    AssertFalse(C > A);
    AssertTrue(G > F);

    return true;
  }

  bool TestMatrixDeterminant() {
    Matrix<double> A({ { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } });
    Matrix<double> B({ { -1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } });
    Matrix<double> C({ { 3, 5, 2 }, { 8, 4, 8 }, { 2, 4, 7 } });
    Matrix<double> D({ { 9, 5, 2, 5 }, { 9, 5, 3, 7 }, { 6, 5, 4, 8 }, { 1, 5, 3, 7 } });
    Matrix<double> E({ { 1, 2 }, { 3, 4 } });
    assert(A.Determinant() == 1.0);
    assert(B.Determinant() == -1.0);
    assert(C.Determinant() == -164);
    assert(D.Determinant() == -40);
    assert(E.Determinant() == -2);
    return true;
  }

  bool TestMatrixTranspose() {
    // Vector
    Matrix<double> vec({ { 1, 2, 3 } });
    Matrix<double> vec2 = vec.Transpose();
    assert(vec2.columns() == vec.rows());
    assert(vec2.rows() == vec.columns());
    for(size_t i = 0; i < 3; ++i) { AssertEqual(vec(0, i), vec2(i, 0)); }
    // Matrix
    Matrix<double> A(1, 2, 2);
    AssertEqual(A, A.Transpose());
    Matrix<double> B({ { 1, 1 }, { 2, 2 } });
    Matrix<double> C({ { 1, 2 }, { 1, 2 } });

    AssertEqual(B.Transpose(), C);

    Matrix<double> D({ { 1, 1 }, { 2, 2 }, { 3, 3 } });
    Matrix<double> E({ { 1, 2, 3 }, { 1, 2, 3 } });
    AssertEqual(D.Transpose(), E);
    return true;
  }

  bool TestMatrixInit() {
    Matrix<float> matrix;
    matrix.Resize(2, 2);
    assert(matrix.rows() == 2);
    assert(matrix.columns() == 2);

    Matrix<double> m2({ { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } });
    assert(m2.rows() == 2);
    assert(m2.columns() == 11);

    Matrix<double> m3({ { 1 }, { 1 } });
    assert(m3.rows() == 2);
    assert(m3.columns() == 1);

    Matrix<double> m5({ { 1.0, 1.0 }, { 1.0, 1.0 } });
    for(size_t i = 0; i < 2; i++) {
      for(size_t j = 0; j < 2; j++) { assert(m5(i, j) == 1.0); }
    }
    Matrix<double> m6(1.0, 2, 2);
    for(size_t i = 0; i < 2; i++) {
      for(size_t j = 0; j < 2; j++) { assert(m6(i, j) == 1.0); }
    }

    Matrix<double> foo = { { 1., 2., 3. } };
    Matrix<float> bar  = foo;
    Matrix<int> baz    = bar;

    AssertEqual(foo(0, 1), 2.0);
    AssertEqual(bar(0, 1), 2.0f);
    AssertEqual(baz(0, 2), 3);

    AssertEqual(foo, Matrix<double>(&foo(0, 0), 3));
    AssertEqual(foo, Matrix<double>((double*)&*foo, 3));

    const Matrix<double> const_foo = foo;
    AssertEqual(const_foo, Matrix<double>(&const_foo(0, 0), 3));
    AssertEqual(const_foo, Matrix<double>((double*)&*const_foo, 3));

    return true;
  }

  bool TestWhere() {
    Matrix<double> A({ { 0, 0 }, { 1, 1 }, { 0, 0 } });
    std::function<bool(double)> cond = [](double i) { return bool(i == 1); };
    auto B = where(cond, A, { { 1, 1 }, { 1, 1 }, { 1, 1 } }, { { 0, 0 }, { 0, 0 }, { 0, 0 } });
    //        std::cout << B;
    assert(B == A);
    return true;
  }

  bool TestGetIndex() {
    Matrix<bool> A(false, 5, 5);
    AssertEqual(A.GetIndex(4, 4), 24);
    AssertEqual(A.GetIndex(0, 0), 0);
    AssertEqual(A.GetIndex(2, 2), 12);
    AssertEqual(A.GetIndex(4, 0), 20);
    AssertEqual(A.GetIndex(4, 2), 22);

    Matrix<double> B({ { 1, 1 }, { 1, 1 } });
    AssertEqual(B.GetIndex(0, 0), 0);
    AssertEqual(B.GetIndex(0, 1), 1);
    AssertEqual(B.GetIndex(1, 0), 2);
    AssertEqual(B.GetIndex(1, 1), 3);

    Matrix<double> C({ { { 1, 1 }, { 1, 1 } }, { { 1, 1 }, { 1, 1 } } });
    AssertEqual(C.GetIndex(0, 0, 0), 0);
    AssertEqual(C.GetIndex(0, 0, 1), 1);
    AssertEqual(C.GetIndex(0, 1, 0), 2);
    AssertEqual(C.GetIndex(0, 1, 1), 3);
    AssertEqual(C.GetIndex(1, 0, 0), 4);
    AssertEqual(C.GetIndex(1, 0, 1), 5);
    AssertEqual(C.GetIndex(1, 1, 0), 6);
    AssertEqual(C.GetIndex(1, 1, 1), 7);
    //        assert(A.GetIndex(5, 2)); //-> Index out of bounds
    return true;
  }

  bool TestArgMinMax() {
    Matrix<int> A({ { 2, 3, 2, 2 }, { 2, 2, 1, 2 } });
    assert(argmin(A) == 6);
    assert(argmax(A) == 1);
    return true;
  }

  bool TestMax() {
    Matrix<int> A({ { 1, 2, 3, 4, 5, 6 } });
    AssertEqual(elemMax(A, 0), 6);
    return true;
  }

  bool TestCorr() {
    Matrix<int> A({ { 1, 2, 3, 4, 5 } });
    Matrix<int> B({ { 5, 4, 3, 2, 1 } });

    assert(Corr(A, B) == 1);

    return true;
  }

  bool TestRandom() {
    Matrix<double> A = Matrix<double>::Random(2, 2, 1, -100, 100);
    Matrix<double> B(0, 2, 2);

    assert(Corr(A, B) < 4);

    Matrix<double> C = Matrix<double>::Random(2, 2, 1);
    assert(Corr(C, B) < 4);

    return true;
  }

  bool TestFromVPtr() {
    float in[4] = { 1, 2, 3, 4 };
    auto A      = from_vptr(in, { 2, 2 });

    Matrix<float> B({ { 1, 2 }, { 3, 4 } });
    assert(A == B);

    return true;
  }

  bool TestVectorAccess() {
    Matrix<int> A({ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
    auto B = A(0);
    Matrix<int> C({ { 1, 1, 1 } });

    assert(B == C);
    return true;
  }

  bool TestGetSlice() {
    Matrix<int> A({ { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
    Matrix<int> B({ { 2, 2 }, { 3, 3 } });

    AssertEqual(A.GetSlice(1, 2, 1, 2), B);
    AssertEqual(A.GetSlice(1, 2, 1), B);

    Matrix<int> C({ { 1, 2 }, { 3, 4 } });
    Matrix<int> D({ { 1, 2 } });
    Matrix<int> E({ { 3, 4 } });

    //        std::cout << C(0) << C(1);
    AssertEqual(C.GetSlice(0, 0), D);
    AssertEqual(C.GetSlice(0, 0, 0), D);
    AssertEqual(C.GetSlice(0, 0, 0, 1), D);
    AssertEqual(C(0), D);
    AssertEqual(C.GetSlice(1, 1), E);
    AssertEqual(C.GetSlice(1, 1, 0), E);
    AssertEqual(C.GetSlice(1, 1, 0, 1), E);
    AssertEqual(C(1), E);
    return true;
  }

  bool TestGetComponents() {
    Matrix<double> A({ { 1, 2, 3, 4, 5, 6, 7, 8, 9 } });

    AssertEqual(A.GetComponents(0), A);

    return true;
  }
  bool TestApply() {
    Matrix<double> A = eye(2, 2);
    Matrix<double> B = eye(2, 2);
    Matrix<double> D = 2 * eye(2, 2);
    Matrix<double> F = 4 * eye(2, 2);

    auto C = A.Apply([](double val) { return val * 2; });
    auto E = D.Apply([](double val) { return val * val; });

    AssertEqual(A, B);
    AssertEqual(C, D);
    AssertEqual(E, F);
    AssertEqual(2 * D, E);

    return true;
  }

  bool TestUnique() {
    Matrix<int> A = Matrix({ { 0, 1 }, { 0, 2 }, { 1, 1 }, { 1, 0 }, { 0, 1 } });
    Matrix<int> B = Matrix({ { 0, 1 }, { 0, 2 }, { 1, 1 }, { 1, 0 } });
    Matrix<int> C = Matrix({ { 0, 1, 2, 3, 4, 1, 2, 3, 4, 5 } }).Transpose();
    Matrix<int> D = Matrix({ { 0, 1, 2, 3, 4, 5 } }).Transpose();

    AssertEqual(unique(A), B);
    AssertEqual(unique(C), D);


    return true;
  }

  bool TestMean() {
    Matrix<double> A = { { 0., 1. }, { 0., 2. }, { 1., 1. }, { 1., 0. }, { 0., 1. } };
    Matrix<double> B = { { 0.7 } };
    Matrix<double> C = { { 0.4, 1.0 } };
    Matrix<double> D = { { 0.5 }, { 1.0 }, { 1.0 }, { 0.5 }, { 0.5 } };

    AssertEqual(mean(A), B);
    AssertEqual(mean(A, 0), C);
    AssertEqual(mean(A, 1), D);

    AssertEqual(elemMean(A, 0), B(0, 0));

    return true;
  }

public:
  void run() override {
    TestMatrixInit();
    TestMatrixMultiplication();
    TestMatrixDivision();
    TestMatrixAddition();
    TestMatrixCompare();
    TestMatrixDeterminant();
    TestMatrixTranspose();
    TestHadamardMultiplication();
    TestKroneckerMultiplication();
    TestHorizontalConcat();
    TestWhere();
    TestGetIndex();
    TestArgMinMax();
    TestCorr();
    TestRandom();
    TestFromVPtr();
    TestVectorAccess();
    TestGetSlice();
    TestApply();
    TestUnique();
    TestMean();
    TestMax();
    TestGetComponents();
  }
};

int main() {
  MatrixTestCase().run();
  return 0;
}
