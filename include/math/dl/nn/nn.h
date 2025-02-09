#pragma once

#include "./engine.h"

template<typename T>
class Module
{
public:
  Module(){}

  virtual std::vector<std::shared_ptr<Value<T>>> parameters();

  void zero_grad(){
    for(auto param : parameters()){
      param->grad = static_cast<T>(0.0f);
    }
  }
};

template<typename T>
class Neuron
: public Module<T>
{
  std::vector<std::shared_ptr<Value<T>>> w;
  std::shared_ptr<Value<T>> b;
public:
  Neuron(size_t num_inputs, int seed=420){
    // set seed
    for(size_t i = 0; i < num_inputs; i++){
      w.push_back(static_cast<T>(0.0));
    }
    b = static_cast<T>(0.0);
  }

  T operator()(const std::vector<T>& x){
    Value<T> act(0.0);
    size_t i = 0;
    for(auto xi : x){
      act += (*w[i].get() * xi) + b;
      i += 1;
    }
    return act.tanh();
  }
};
