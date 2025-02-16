#include "../../Test.h"
#include <cstdlib>
#include <math/dl/nn/nn.h>
#include <numeric>
#include <vector>
#include <memory>
#include <sstream>
#include <string>


class NeuronTestCase : public Test {

  bool TestOperator(){
    Neuron<float> NN(1);

    std::vector<std::vector<std::shared_ptr<Value<float>>>> X = {
      {std::make_shared<Value<float>>(1.0f),},
      {std::make_shared<Value<float>>(2.0f),},
      {std::make_shared<Value<float>>(3.0f),},
      {std::make_shared<Value<float>>(4.0f),},
    };

    auto out = NN(X);

    std::cout << NN << std::endl;
    std::cout << *out[0][0] << std::endl;
    AssertEqual(out.size(), 4);
    for(auto elem : out){
      AssertEqual(elem.size(), 1);
    }

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
    Layer<float> layer(1, 2);
    std::vector<std::vector<std::shared_ptr<Value<float>>>> X = {
      {std::make_shared<Value<float>>(1.0f),},
      {std::make_shared<Value<float>>(2.0f),},
      {std::make_shared<Value<float>>(3.0f),},
      {std::make_shared<Value<float>>(4.0f),},
    };

    auto out = layer(X);
    AssertEqual(out.size(), 4);
    for(auto elem : out){
      AssertEqual(elem.size(), 2);
    }

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
    std::vector<size_t> sizes = {5, 10, 1};
    MLP<float> mlp(1, sizes);
    std::vector<std::vector<std::shared_ptr<Value<float>>>> X = {
      {std::make_shared<Value<float>>(1.0f),},
      {std::make_shared<Value<float>>(2.0f),},
      {std::make_shared<Value<float>>(3.0f),},
      {std::make_shared<Value<float>>(4.0f),},
    };

    auto out = mlp(X);

    out[0][0]->backward();

    AssertEqual(out.size(), 4);
    for(auto elem : out){
      AssertEqual(elem.size(), 1);
    }

    //std::cout << *out[0][0] << std::endl;
    //std::cout << mlp << std::endl;

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
    std::vector<size_t> hidden = {1};
    auto n = MLP<float>(3, hidden);

    size_t EPOCH = 500;
    float learning_rate = 0.01;

    for(size_t epoch = 0; epoch < EPOCH; ++epoch){
      n.zero_grad();
      std::vector<std::vector<std::shared_ptr<Value<float>>>> ypred = n(xs);
      std::cout << "YPRED:"<<std::endl;
      size_t ix = 0;
      size_t iy = 0;
      for(auto pred : ypred){
        std::cout << iy++ << ": " << std::endl;
        for(auto pi : pred){
          std::cout << "\t\t" << *ys[ix++] << " " << *pi << std::endl;
          if(pi->data > 0.0){
            std::cout << "\t\t\t\t\bCHANGE"<<std::endl; 
          }
        }
      }
      auto l = std::make_shared<Value<float>>(0.0f);
      for(size_t i = 0; i < ypred.size(); ++i){
        l += (ypred[i][0] - ys[i])->pow(2.0f);
        std::cout << "LO: " << *l << std::endl;
      }

      l->backward();

      l->printGraph(l);

      auto lr = (1.0 - (1.0 - learning_rate) * epoch / EPOCH);
      std::cout << "\033[31m[";
      for(auto param : n.parameters()){
        param->data += -lr * param->grad;
        std::cout << param->data << "{" << param->grad << "} | ";
      }
      std::cout << "]\033[0m" << std::endl;

      std::cout << "\033[32mEPOCH " << epoch  << ": loss = " << *l << "; LR: " << lr << "\033[0m" << std::endl;
      ypred.clear();
      l.reset();
    }

    return true;
  }

  void run(){
    TestOperator();
    TestTrain();
  }

};


class MNISTTest: public Test {
  void read_file(std::vector<std::vector<std::shared_ptr<Value<float>>>>& values, std::vector<std::shared_ptr<Value<float>>>& labels){
    FILE* fp = fopen("../../assets/mnist/tiny.csv", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    std::cout << "READING FILE!" << std::endl;
    char* line = NULL;
    size_t len = 0;
    while ((getline(&line, &len, fp)) != -1) {
      // using printf() in all tests for consistency
      std::string l = std::string(line);
      std::shared_ptr<Value<float>> label;
      std::vector<std::shared_ptr<Value<float>>> vals;
      while(l.length() > 0){
        auto nextId = l.find(",");
        if(nextId == std::string::npos){
          if(l.length() > 0){
            vals.push_back(std::make_shared<Value<float>>(atoi(l.c_str())/255.));
          }
          // next line please
          break;
        }
        auto val = l.substr(0, nextId);
        if(label == nullptr){
          label = std::make_shared<Value<float>>(atoi(val.c_str()));
        } else {
          vals.push_back(std::make_shared<Value<float>>(atoi(val.c_str())/255.));
        }
        // +1 for ","
        l.erase(0, nextId + 1);
      }
      labels.push_back(label);
      values.push_back(vals);
    }
    fclose(fp);
    if (line)
      free(line);

    std::cout << "\033[35mFound: " << values.size() 
      << " Values and " << labels.size()
      << " Labels" << "\033[0m" << std::endl;

    if(values.size() <= 50 && values[0].size() < 10){

      std::cout << "\033[35mLabels: ";
      for(auto label : labels){
        std::cout << *label << " | ";
      }
      std::cout << "\033[0m" << std::endl;
      
      std::cout << "\033[34mValues: \n";
      for(auto vals : values){
        for(auto val : vals){
          std::cout << *val << " | ";
        }
        std::cout << std::endl;
      }
      std::cout << "\033[0m" << std::endl;
    }
  }
public:
  void run(){
    std::vector<std::shared_ptr<Value<float>>> labels;
    std::vector<std::vector<std::shared_ptr<Value<float>>>> values;
    read_file(values, labels);

    std::vector<size_t> hidden = {512, 32, 1};
    auto n = MLP<float>(values[0].size(), hidden);

    size_t EPOCH = 5;
    float learning_rate = 0.01;

    int batch_size = 1;

    size_t num_shards = ((values.size() + (values.size() % batch_size)) / batch_size);

    for(size_t epoch = 0; epoch < EPOCH; ++epoch){
      int idx = static_cast<int>(Random::Get(0, num_shards));

      std::vector<std::vector<std::shared_ptr<Value<float>>>>::const_iterator first = values.begin() + (idx * batch_size);
      std::vector<std::vector<std::shared_ptr<Value<float>>>>::const_iterator last = values.begin() + ((idx + 1) * batch_size);
      std::vector<std::vector<std::shared_ptr<Value<float>>>> local_values(first, last);

      std::vector<std::shared_ptr<Value<float>>>::const_iterator first_l = labels.begin() + (idx * batch_size);
      std::vector<std::shared_ptr<Value<float>>>::const_iterator last_l = labels.begin() + ((idx + 1) * batch_size);
      std::vector<std::shared_ptr<Value<float>>> local_labels(first_l, last_l);

      std::vector<std::vector<std::shared_ptr<Value<float>>>> ypred = n(local_values);
      //std::cout << "YPRED:"<<std::endl;
      std::vector<std::shared_ptr<Value<float>>> losses;
      for(size_t i = 0; i < ypred[0].size(); ++i){
        auto lo = (ypred[0][i] - local_labels[i])->pow(2.0f);
        //std::cout << "LO: " << *lo << std::endl;
        losses.push_back(lo);
      }
      auto l = std::make_shared<Value<float>>(0.0f, false);
      auto loss = std::accumulate(losses.begin(), losses.end(), l);

      n.zero_grad();
      loss->backward();

      auto lr = (1.0 - (1.0 - learning_rate) * epoch / EPOCH);
      //std::cout << "\033[31m[";
      for(auto param : n.parameters()){
        param->data += -lr * param->grad;
        //std::cout << param->data << "{" << param->grad << "} | ";
      }
      //std::cout << "]\033[0m" << std::endl;

      std::cout << "\033[32mEPOCH " << epoch  << ": loss = " << *loss << "; LR: " << lr << "\033[0m" << std::endl;
      local_values.clear();
      ypred.clear();
    }
  }

};


int main() {
  NeuronTestCase().run();
  LayerTestCase().run();
  MLPTestCase().run();
  MNISTTest().run();
  return 0;
}


