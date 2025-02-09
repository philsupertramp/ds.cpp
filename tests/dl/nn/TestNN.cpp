#include "../../Test.h"
#include <math/dl/nn/nn.h>
#include <vector>


class NeuronTestCase : public Test {

  bool TestOperator(){
    Neuron<float> nn(10);

    std::vector<float> vals = {1.f,2.f,3.f,4.f};
    auto foo = nn(vals);

    return true;
  }
public:
  void run(){

  }
};

int main() {
  NeuronTestCase().run();
  return 0;
}


