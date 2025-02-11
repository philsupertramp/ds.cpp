#pragma once

#include "./engine.h"
#include "../../Random.h"
#include <memory>

template<typename T>
std::vector<std::shared_ptr<Value<T>>> create_ptrs(const std::vector<T>& x){
  std::vector<std::shared_ptr<Value<T>>> xx;
  for(auto &&e : x){
    auto p = std::make_shared<Value<T>>(e);
    if(p) xx.push_back(p);
  }
  return xx;
}

template<typename T>
class Module
{
public:
  Module(){}


  Value<T> operator()(const std::vector<T>& x){
    return this(create_ptrs(x));
  }

  void zero_grad(){
    for(auto param : parameters()){
      param->grad = static_cast<T>(0.0f);
    }
  }

  virtual std::vector<std::vector<std::shared_ptr<Value<T>>>> operator()(const std::vector<std::vector<std::shared_ptr<Value<T>>>>& X) = 0;
  virtual std::vector<std::shared_ptr<Value<T>>> parameters() = 0;
};

template<typename T>
class Neuron
: public Module<T>
{
  std::vector<std::shared_ptr<Value<T>>> w;
  std::shared_ptr<Value<T>> b;
public:
  Neuron(size_t num_inputs, int seed=420)
  {
    Random::SetSeed(seed);
    // set seed
    for(size_t i = 0; i < num_inputs; i++){
      w.push_back(std::make_shared<Value<T>>(static_cast<T>(Random::Get())));
    }
    b = std::make_shared<Value<T>>(static_cast<T>(1.0));
  }

  std::vector<std::vector<std::shared_ptr<Value<T>>>> operator()(const std::vector<std::vector<std::shared_ptr<Value<T>>>>& X) override {
    std::vector<std::shared_ptr<Value<T>>> out;

    for(auto xx : X){
      auto act = std::make_shared<Value<T>>(0.0f);
      size_t i = 0;
      DEBUG_PRINT("casting input...\n");
      
      DEBUG_PRINT("Input converted\n");
      for(auto p : xx){
        DEBUG_PRINT("Elem %f -> %f\n", p->data, act->data);
        act = act + ((w[i] * p) + b);
        i += 1;
      }
      auto final = (*act).tanh();
      out.push_back(final);
    }
    return {out};
  }

  std::vector<std::shared_ptr<Value<T>>> parameters() override {
    std::vector<std::shared_ptr<Value<T>>> params;
    for(auto elem : w){
      params.push_back(elem);
    }
    params.push_back(b);
    return params;
  }

  const std::vector<std::shared_ptr<Value<T>>> parameters() const {
    std::vector<std::shared_ptr<Value<T>>> params;
    for(auto elem : w){
      params.push_back(elem);
    }
    params.push_back(b);
    return params;
  }

  template<typename C>
  friend std::ostream& operator<<(std::ostream& ostr, const Neuron<C>& val);
};

template<typename T>
std::ostream& operator<<(std::ostream& ostr, const Neuron<T>& val){
  auto params = val.parameters();
  ostr << "[";
  for(auto param : params){
    ostr << *param << ", ";
  }
  ostr << "]";
  return ostr;
}


template<typename T>
class Layer
: public Module<T>
{
  std::vector<std::shared_ptr<Neuron<T>>> neurons;
public:
  Layer(size_t num_inputs, size_t num_outputs, int seed=420)
  {
    for(size_t i = 0; i < num_outputs; i++){
      neurons.push_back(std::make_shared<Neuron<T>>(num_inputs, seed));
    }
  }

  std::vector<std::vector<std::shared_ptr<Value<T>>>> operator()(const std::vector<std::vector<std::shared_ptr<Value<T>>>>& X) override {
    std::vector<std::vector<std::shared_ptr<Value<T>>>> out;
    for(auto neuron : neurons){
      out.push_back((*neuron)(X)[0]);
    }
    return out;
  }
  std::vector<std::shared_ptr<Value<T>>> parameters() {
    std::vector<std::shared_ptr<Value<T>>> params;
    for(auto neuron : neurons){
      auto p = neuron->parameters();
      params.insert(params.end(), p.begin(), p.end());
    }
    return params;
  }
  const std::vector<std::shared_ptr<Value<T>>> parameters() const {
    std::vector<std::shared_ptr<Value<T>>> params;
    for(auto neuron : neurons){
      auto p = neuron->parameters();
      params.insert(params.end(), p.begin(), p.end());
    }
    return params;
  }
  template<typename C>
  friend std::ostream& operator<<(std::ostream& ostr, const Layer<C>& val);

};

template<typename C>
std::ostream& operator<<(std::ostream& ostr, const Layer<C>& val){
  for(auto n : val.neurons){
    ostr << *n << ", ";
  }
  return ostr;
}


template<typename T>
class MLP
: public Module<T>
{
  std::vector<std::shared_ptr<Layer<T>>> layers;
public:
  MLP(size_t num_inputs, const std::vector<size_t>& hidden, int seed=420)
  {
    std::vector<size_t> layerSizes = {num_inputs};
    layerSizes.insert(layerSizes.end(), hidden.begin(), hidden.end());
    for(size_t i = 0; i < hidden.size(); ++i){
      layers.push_back(std::make_shared<Layer<T>>(layerSizes[i], layerSizes[i+1], seed));
    }
  }
  std::vector<std::vector<std::shared_ptr<Value<T>>>> operator()(const std::vector<std::vector<std::shared_ptr<Value<T>>>>& X) override {
    std::vector<std::vector<std::shared_ptr<Value<T>>>> out = X;
    for(auto layer : layers){
      out = (*layer)(out);
    }
    return out;
  }
  std::vector<std::shared_ptr<Value<T>>> parameters() {
    std::vector<std::shared_ptr<Value<T>>> params;
    for(auto layer : layers){
      auto p = layer->parameters();
      params.insert(params.end(), p.begin(), p.end());
    }
    return params;
  }
  const std::vector<std::shared_ptr<Value<T>>> parameters() const {
    std::vector<std::shared_ptr<Value<T>>> params;
    for(auto layer : layers){
      auto p = layer->parameters();
      params.insert(params.end(), p.begin(), p.end());
    }
    return params;
  }
  template<typename C>
  friend std::ostream& operator<<(std::ostream& ostr, const MLP<C>& val);

};

template<typename C>
std::ostream& operator<<(std::ostream& ostr, const MLP<C>& val){
  ostr << "MLP(";
  for(auto layer : val.layers){
    ostr << layer << std::endl;
  }
  ostr << ")";
  return ostr;
}

