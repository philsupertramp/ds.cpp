#include "../../Test.h"
#include <math/dl/nn/nn.h>
#include <vector>
#include <memory>


class NeuronTestCase : public Test {

  bool TestOperator(){
    Neuron<float> NN(10);

    std::vector<std::vector<std::shared_ptr<Value<float>>>> X = {
      {std::make_shared<Value<float>>(1.0f),},
      {std::make_shared<Value<float>>(2.0f),},
      {std::make_shared<Value<float>>(3.0f),},
      {std::make_shared<Value<float>>(4.0f),},
    };

    auto out = NN(X);

    std::cout << NN << std::endl;
    std::cout << *out[0][0] << std::endl;

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
    std::vector<std::vector<std::shared_ptr<Value<float>>>> X = {
      {std::make_shared<Value<float>>(1.0f),},
      {std::make_shared<Value<float>>(2.0f),},
      {std::make_shared<Value<float>>(3.0f),},
      {std::make_shared<Value<float>>(4.0f),},
    };

    auto out = layer(X);

    std::cout << *out[0][0] << std::endl;
    std::cout << layer << std::endl;

    return true;

  }

  void run(){
    TestOperator();
  }

};
class MLPTestCase
: public Test
{
public:

  bool TestOperator(){
    std::vector<size_t> sizes = {5, 1};
    MLP<float> mlp(1, sizes);
    std::vector<std::vector<std::shared_ptr<Value<float>>>> X = {
      {std::make_shared<Value<float>>(1.0f),},
      {std::make_shared<Value<float>>(2.0f),},
      {std::make_shared<Value<float>>(3.0f),},
      {std::make_shared<Value<float>>(4.0f),},
    };

    auto out = mlp(X);

    out[0][0]->backward();

    std::cout << *out[0][0] << std::endl;
    std::cout << mlp << std::endl;

    return true;

  }

  bool TestTrain(){

    auto value = [](float val){ return std::make_shared<Value<float>>(val); };

    std::vector<std::vector<std::shared_ptr<Value<float>>>> xs = {
      {value(2), value(3), value(-1)},
      {value(3), value(-1), value(0.5)},
      {value(0.5), value(1), value(1)},
      {value(1), value(1), value(-1)},
    };

    std::vector<std::shared_ptr<Value<float>>> ys = {value(1), value(-1), value(-1), value(1)};
    std::vector<size_t> hidden = {4, 4, 1};
    auto n = MLP<float>(3, hidden);

    size_t EPOCH = 300;
    float learning_rate = 0.01;

    for(size_t epoch = 0; epoch < EPOCH; ++epoch){
      std::vector<std::vector<std::shared_ptr<Value<float>>>> ypred;
      for(auto x : xs){
        ypred.push_back(n({x})[0]);
      }
    std::cout << "YPRED:"<<std::endl;
    size_t ix =0;
    for(auto pred : ypred){
         for(auto pi : pred){
              std::cout << ix++ << *pi;
         }
         std::cout<<std::endl;
    }
      auto loss = std::make_shared<Value<float>>(0.0);
      for(size_t i = 0; i < ypred.size(); ++i){
        loss = loss + (ypred[i][0] - ys[i])->pow(2.0f);
      }

      n.zero_grad();
      loss->backward();

      auto lr = -(1.0 - (1.0 - learning_rate) * epoch / EPOCH);
      std::cout << "[\033[31m";
      for(auto param : n.parameters()){
        param->data += param->grad * lr;
        std::cout << param->grad << " | ";
      }
      std::cout << "]\033[0m" << std::endl;

      std::cout << "\033[32mEPOCH " << epoch  << ": loss = " << *loss << "; LR: " << lr << "\033[0m" << std::endl;
    }

    return true;
  }

  void run(){
    TestOperator();
    TestTrain();
  }

};

int main() {
  NeuronTestCase().run();
  LayerTestCase().run();
  MLPTestCase().run();
  return 0;
}


