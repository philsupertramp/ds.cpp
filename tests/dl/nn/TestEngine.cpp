#include "../../Test.h"
#include <math/dl/nn/engine.h>
#include <math/Random.h>
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
    AssertEqual(foo->left->grad, 2.0f);
    AssertEqual(foo->right->grad, 2.0f);

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
    AssertEqual(foo->left->grad, 16.0f);
    AssertEqual(foo->right->grad, 2.0f);
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

    AssertEqual(foo->grad, 1.0f);
    AssertEqual(foo->left->grad, 1.0f);
    AssertEqual(foo->right->grad, -1.0f);
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

    foo->backward();

    AssertEqual(foo->grad, 1.0f);
    AssertEqual(foo->left->grad, 1.0f/2.);
    AssertEqual(foo->right->grad, -1.0f);

    return true;
  }

  bool TestPow(){
    float data = 2.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->pow(4.0);

    AssertEqual(foo->data, /* 2^16 */ 16.f);
    AssertEqual((int)foo->op, (int)OP::OP_POW);

    foo->backward();

    std::cout << "POW: " << *foo << std::endl << *(foo->left) << std::endl;

    AssertEqual(foo->grad, 1.0f);
    AssertEqual(foo->left->grad, 32.0f);

    return true;
  }

  bool TestExp(){
    float data = 5.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->exp();

    AssertEqual(foo->data, /* exp(5.0) */ 148.413162f);


    foo->backward();

    AssertEqual(foo->grad, 1.0f);
    AssertEqual(foo->left->grad, 148.413162f);

    return true;
  }

  bool TestTanH(){
    float data = 5.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->tanh();

    AssertEqual(foo->data, /* tanh(5.0) */ 0.9999092f);

    foo->backward();

    AssertEqual(foo->grad, 1.0f);
    AssertEqual(foo->left->grad, 0.00018154751199999999);

    return true;
  }

  bool TestReLU(){
    float data = 5.0f;
    auto val = std::make_shared<Value<float>>(data);

    auto foo = val->relu();
    auto bar = (val * -1.f)->relu();

    AssertEqual(foo->data, 5.0f);
    AssertEqual(bar->data, 0.0f);

    foo->backward();

    AssertEqual(foo->grad, 1.0f);
    AssertEqual(foo->left->grad, 1.0f);

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
    std::cout << "Computing D = C->ReLU() [" << *c << ": " << c->left << " & " << c->right <<"]" << std::endl;
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

  bool TestExample(){
    
    std::vector<std::shared_ptr<Value<float>>> vals = {
      std::make_shared<Value<float>>(0.f),
      std::make_shared<Value<float>>(2.f),
      std::make_shared<Value<float>>(3.5f),
      std::make_shared<Value<float>>(5.f),
      std::make_shared<Value<float>>(12.4f),
      std::make_shared<Value<float>>(13.f),
    };
    std::vector<std::shared_ptr<Value<float>>> labels = {
      std::make_shared<Value<float>>(10.f),
      std::make_shared<Value<float>>(15.f),
      std::make_shared<Value<float>>(17.5f),
      std::make_shared<Value<float>>(19.5f),
      std::make_shared<Value<float>>(21.13f),
      std::make_shared<Value<float>>(25.f),
    };

    std::shared_ptr<Value<float>> W = std::make_shared<Value<float>>((float)0.1);
    std::shared_ptr<Value<float>> b = std::make_shared<Value<float>>((float)1.0);

    std::shared_ptr<Value<float>> val;

    for(auto x : vals){
      val = (x * x) * W + b;
      AssertEqual(val->data, (x->data * x->data) * W->data + b->data);
    }

    auto x = vals[0];

    val = x * W + b;
    val->backward();

    std::cout << "\033[31mOP LEFT";
    val->printGraph(val);
    std::cout << "\033[0m;" << std::endl;

    auto first_grad = val->grad;

    AssertEqual(val->right, b);
    AssertEqual(val->left->right, W);
    AssertEqual(val->left->left, x);

    val = b + x * W;
    val->backward();
    std::cout << "\033[31mOP RIGHT";
    val->printGraph(val);
    std::cout << "\033[0m;" << std::endl;

    AssertEqual(val->right->left, x);
    AssertEqual(val->right->right, W);
    AssertEqual(val->left, b);

    AssertEqual(first_grad, val->grad);

    val = (x * x) * W + b;

    val->printGraph(val);

    AssertEqual(val->right, b);
    AssertEqual(val->left->right, W);
    AssertEqual(val->left->left->left, x);
    AssertEqual(val->left->left->right, x);

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

    TestExample();
  }
};

int main() {
  ValueTestCase().run();
  return 0;
}


