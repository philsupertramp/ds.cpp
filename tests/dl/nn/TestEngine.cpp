#include "../../Test.h"
#include <math/dl/nn/engine.h>


class ValueTestCase : public Test {
public:
  bool TestAddition(){
    float data = 1.0f;
    auto val = Value<float>(&data);

    auto foo = val + val;

    AssertEqual(*(float*)foo.data, 2.0f);

    AssertEqual((int)foo.op, (int)OP::OP_ADD);
    return true;
  }

  bool TestMultiplication(){
    float data = 2.0f;
    auto val = Value<float>(&data);
    float data2 = 16.0f;
    auto val2 = Value<float>(&data2);

    auto foo = val * val2;

    AssertEqual(*(float*)foo.data, 16.0 * 2.0);
    AssertEqual((int)foo.op, (int)OP::OP_MUL);
    return true;
  }

  bool TestSubtraction(){
    float data = 2.0f;
    auto val = Value<float>(&data);
    float data2 = 1.0f;
    auto val2 = Value<float>(&data2);

    auto foo = val - val2;

    AssertEqual(*(float*)foo.data, 1.0f);
    AssertEqual((int)foo.op, (int)OP::OP_SUB);
    return true;
  }

  bool TestDivision(){
    float data = 4.0f;
    float data2 = 2.0f;

    auto val = Value<float>(&data);
    auto val2 = Value<float>(&data2);

    auto foo = val / val2;

    AssertEqual(*(float*)foo.data, 2.0f);
    AssertEqual((int)foo.op, (int)OP::OP_DIV);

    return true;
  }

  bool TestPow(){
    float data = 2.0f;
    auto val = Value<float>(&data);

    auto foo = val.pow(16.0);

    AssertEqual(*(float*)foo.data, /* 2^16 */ 65536.f);
    AssertEqual((int)foo.op, (int)OP::OP_POW);
    return true;
  }

  bool TestExp(){
    float data = 5.0f;
    auto val = Value<float>(&data);

    auto foo = val.exp();

    AssertEqual(*(float*)foo.data, /* exp(5.0) */ 148.413162f);

    return true;
  }

  bool TestTanH(){
    float data = 5.0f;
    auto val = Value<float>(&data);

    auto foo = val.tanh();

    AssertEqual(*(float*)foo.data, /* tanh(5.0) */ 0.9999092f);

    return true;
  }

  bool TestReLU(){
    float data = 5.0f;
    auto val = Value<float>(&data);

    auto foo = val.relu();
    auto bar = (val * -1.f).relu();

    AssertEqual(*(float*)foo.data, 5.0f);
    AssertEqual(*(float*)bar.data, 0.0f);

    return true;
  }


  bool TestBackward(){
    std::cout << "START TEST BACKWARD" << std::endl;
    float data = 1.0f;
    float data2 = 1.0f;

    auto val = Value<float>(&data);
    auto val2 = Value<float>(&data2);

    auto res = val + val2;

    res.backward();

    AssertEqual(res.grad, 1.0f);

    data = 10.0f;
    data2 = 2.0f;

    val = Value<float>(&data);
    val2 = Value<float>(&data2);

    std::cout << "MULTI" << std::endl;
    res = val * val2;
    std::cout << "END MULTI" << std::endl;
    std::cout << "ADD" << std::endl;
    auto res2 = res + val2;
    std::cout << "END ADD" << std::endl;

    std::cout << "START BACKWARD " << val << " " << val2 << std::endl;
    res2.backward();
    std::cout << "END BACKWARD " << res  << std::endl << res2 << std::endl;

    AssertEqual(*(float*)res.data, 22.0f);
    AssertEqual(res.grad, 1.0f);
    AssertEqual(val2.grad, 10.0f);

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


