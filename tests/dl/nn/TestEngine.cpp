#include "../../Test.h"
#include <math/dl/nn/engine.h>
#include <memory>


class ValueTestCase : public Test {
public:
  bool TestAddition(){
    float data = 1.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val + val;

    AssertEqual(foo->data, 2.0f);
    AssertEqual((int)foo->op, (int)OP::OP_ADD);

    foo->backward();

    std::cout << *foo << std::endl;

    AssertEqual(foo->grad, 1.0f);
    AssertEqual(foo->prev[0]->grad, 2.0f);
    AssertEqual(foo->prev[1]->grad, 2.0f);

    return true;
  }

  bool TestMultiplication(){
    float data = 2.0f;
    auto val = std::make_shared<Value<float>>(data);
    float data2 = 16.0f;
    auto val2 = std::make_shared<Value<float>>(data2);

    auto foo = val * val2;

    AssertEqual(foo->data, 16.0 * 2.0);
    AssertEqual((int)foo->op, (int)OP::OP_MUL);
    foo->backward();

    AssertEqual(foo->grad, 1.f);
    AssertEqual(foo->prev[0]->grad, 16.0f);
    AssertEqual(foo->prev[1]->grad, 2.0f);


    return true;
  }

  bool TestSubtraction(){
    float data = 2.0f;
    auto val = std::make_shared<Value<float>>(data);
    float data2 = 1.0f;
    auto val2 = std::make_shared<Value<float>>(data2);

    auto foo = val - val2;

    AssertEqual(foo->data, 1.0f);
    AssertEqual((int)foo->op, (int)OP::OP_SUB);
    foo->backward();
    std::cout << *foo << std::endl;

    return true;
  }

  bool TestDivision(){
    float data = 4.0f;
    float data2 = 2.0f;

    auto val = std::make_shared<Value<float>>(data);
    auto val2 = std::make_shared<Value<float>>(data2);

    auto foo = val / val2;

    AssertEqual(foo->data, 2.0f);
    AssertEqual((int)foo->op, (int)OP::OP_DIV);

    return true;
  }

  bool TestPow(){
    float data = 2.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->pow(16.0);

    AssertEqual(foo->data, /* 2^16 */ 65536.f);
    AssertEqual((int)foo->op, (int)OP::OP_POW);
    return true;
  }

  bool TestExp(){
    float data = 5.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->exp();

    AssertEqual(foo->data, /* exp(5.0) */ 148.413162f);

    return true;
  }

  bool TestTanH(){
    float data = 5.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->tanh();

    AssertEqual(foo->data, /* tanh(5.0) */ 0.9999092f);

    return true;
  }

  bool TestReLU(){
    float data = 5.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->relu();
    auto bar = (val * -1.f)->relu();

    AssertEqual(foo->data, 5.0f);
    AssertEqual(bar->data, 0.0f);

    return true;
  }


  bool TestBackward(){
    std::cout << "Creating A[";
    auto a = std::make_shared<Value<float>>(2.0);
    std::cout << *a << "]" << std::endl;
    std::cout << "Creating B[";
    auto b = std::make_shared<Value<float>>(3.0);
    std::cout << *b << "]" << std::endl;
    std::cout << "Computing C = A[" << *a <<"] * B[" << *b << "]" << std::endl;
    auto c = a * b;
    std::cout << "Computing D = C->ReLU() [" << *c << ": " << c->prev[0] << " & " << c->prev[1] <<"]" << std::endl;
    auto d = c->relu();
    std::cout << "Computing D backward() [" << *d << "]" << std::endl;
    d->backward();

    std::cout << *a << " GRAD should be 3.0" << std::endl;
    std::cout << *b << " GRAD should be 2.0" << std::endl;
    std::cout << *c << std::endl;
    std::cout << *d << std::endl;

    AssertEqual(a->grad, 3.0);
    AssertEqual(b->grad, 2.0);
    AssertEqual(c->grad, 1.0);
    AssertEqual(d->grad, 1.0);

    return true;
  }

  void run(){
    TestAddition();
    TestMultiplication();
    TestSubtraction();
    TestDivision();
    TestPow();
    TestExp();
    TestTanH();
    TestReLU();

    TestBackward();
  }
};

int main() {
  ValueTestCase().run();
  return 0;
}


