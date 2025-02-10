#include "../../Test.h"
#include <math/dl/nn/nn.h>
#include <vector>
#include <memory>


class NeuronTestCase : public Test {

  bool TestOperator(){
    Neuron<float> NN(10);

    std::vector<std::shared_ptr<Value<float>>> X = {
      std::make_shared<Value<float>>(1.0f),
      std::make_shared<Value<float>>(2.0f),
      std::make_shared<Value<float>>(3.0f),
      std::make_shared<Value<float>>(4.0f),
    };

    auto out = NN(X);

    std::cout << NN << std::endl;

    return true;
  }

public:
  void run(){
    TestOperator();
  }
};

class LayerTestCase
: public Test
{
public:

  bool TestOperator(){
    Layer<float> layer(10, 2);
    std::vector<std::shared_ptr<Value<float>>> X = {
      std::make_shared<Value<float>>(1.0f),
      std::make_shared<Value<float>>(2.0f),
      std::make_shared<Value<float>>(3.0f),
      std::make_shared<Value<float>>(4.0f),
    };

    auto out = layer(X);

    std::cout << *out << std::endl;
    std::cout << layer << std::endl;

    return true;

  }

  void run(){
    TestOperator();
  }

};

int main() {
  NeuronTestCase().run();
  LayerTestCase().run();
  return 0;
}


