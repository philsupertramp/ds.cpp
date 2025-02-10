#pragma once

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <functional>
#include <vector>
#include <algorithm>

#define DEBUG 1

#if (DEBUG > 0)
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif


enum OP {
  OP_ADD=0,
  OP_MUL,
  OP_SUB,
  OP_DIV,
  OP_POW,
  OP_EXP,
  OP_TANH,
  OP_RELU,
  OP_NONE
};

enum datatype {
  FLOAT_32 = 0,
  TYPE_COUNT
};


template<typename T>
class Value
: public std::enable_shared_from_this<Value<T>>
{
public:
  T data;
  T grad;

  std::vector<std::shared_ptr<Value<T>>> prev;
  OP op;
  
  const char* label;

  std::function<void()> backward_;
  bool is_param = false;
  
  Value(const T data_, bool is_param_=false)
  : std::enable_shared_from_this<Value<T>>()
  {
    data = data_;
    prev = {};
    op = OP::OP_NONE;
    label = "";
    grad = static_cast<T>(0.0f);
    backward_ = [](){};
    is_param = is_param_;
  }

  Value(const T data_, std::vector<std::shared_ptr<Value<T>>> children, OP op_, const char* label_)
  : std::enable_shared_from_this<Value<T>>()
  {
    data = data_;
    prev = children;
    op = op_;
    label = std::move(label_);
    grad = static_cast<T>(0.0);
    backward_ = [](){};
    DEBUG_PRINT("Constructor, NEW OBJECT! %p\n",(void*)this);
  }
  Value(const Value& other)
  : std::enable_shared_from_this<Value<T>>(),
    data(other.data),
    prev(other.prev),
    op(other.op),
    label(std::move(other.label)),grad(other.grad),backward_(other.backward_)
  {
    if(other.prev.size() == 2){
    DEBUG_PRINT("const Value COPY CONSTRUCTOR [%p] %f | %f\n",(void*)this,other.prev[0]->grad,other.prev[1]->grad);
    } else if (other.prev.size() == 1){
    DEBUG_PRINT("const Value COPY CONSTRUCTOR [%p] %f\n",(void*)this,other.prev[0]->grad);
    }
  }
  Value(Value* other)
  : std::enable_shared_from_this<Value<T>>(),
    data(other->data),
    prev(other->prev),
    op(other->op),
    label(std::move(other->label)),
    grad(other->grad),
    backward_(other->backward_)
  {
    if(other->prev.size() == 2){
    DEBUG_PRINT("COPY CONSTRUCTOR %f | %f\n",other->prev[0]->grad,other->prev[1]->grad);
    } else if (other->prev.size() == 1){
    DEBUG_PRINT("COPY CONSTRUCTOR %f\n",other->prev[0]->grad);
    }
  }
  Value(Value&& other) noexcept
  : std::enable_shared_from_this<Value<T>>(),
    data(other.data),
    prev(other.prev),
    op(other.op),
    label(std::move(other.label)),
    grad(other.grad),
    backward_(std::move(other.backward_)) {DEBUG_PRINT("MOVE\n");}

  ~Value() = default;

  // Assignment and comparison

  Value& operator=(Value&& other) noexcept {
    DEBUG_PRINT("ASSIGN\n");
    if (this != &other) {
      data = other.data;
      prev = other.prev;
      op = other.op;
      label = std::move(other.label);
      grad = other.grad;
      backward_ = std::move(other.backward_);
    }
      return *this;
  }

  bool operator==(const Value<T>& rhs){
    DEBUG_PRINT("%p %p\n", *this->data,*rhs->data);
    return (*this->data) == (*rhs.data);
  }

  // Math OPERATORS

  std::shared_ptr<Value<T>> div(const std::shared_ptr<Value<T>>& other){
    auto res = other->pow(-1.) * this->shared_from_this();
    res->op = OP::OP_DIV;
    return res;
  }
  std::shared_ptr<Value<T>> sub(const std::shared_ptr<Value<T>>& other){
    auto neg = std::make_shared<Value<T>>(-1);
    neg = neg * other;
    auto res = add(neg);
    res->op = OP::OP_SUB;
    return res;
  }

  std::shared_ptr<Value<T>> add(const std::shared_ptr<Value<T>>& other){
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this(), other};
    auto val = data + other->data;
    auto out = std::make_shared<Value<T>>(val, children, OP::OP_ADD, "+");

    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT("add %f(%p) * %f\n",out->prev[0]->grad, (void*)out->prev[0].get(),out->prev[1]->grad);
        out->prev[0]->grad += static_cast<T>(1.0) * out->grad;
        out->prev[1]->grad += static_cast<T>(1.0) * out->grad;
        DEBUG_PRINT("add %f(%p) * %f after\n",out->prev[0]->grad, (void*)out->prev[0].get(),out->prev[1]->grad);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> mmul(const std::shared_ptr<Value<T>>& other){
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this(), other};
    //DEBUG_PRINT("MULTI PAST CHILDREN\n");
    auto val = data * other->data;
    auto out = std::make_shared<Value<T>>(val, children, OP::OP_MUL, "*");
    //DEBUG_PRINT("MULTI PAST CONSTRUCTOR\n");
    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT("mul %f(%p) %f(%p) * %f\n",out->grad,(void*)out.get(),out->prev[0]->grad, (void*)out->prev[0].get(),out->prev[1]->grad);
        out->prev[0]->grad += out->prev[1]->data * out->grad;
        out->prev[1]->grad += out->prev[0]->data * out->grad;
        DEBUG_PRINT("mul %f(%p) * %f after\n",out->prev[0]->grad, (void*)out->prev[0].get(),out->prev[1]->grad);
      };
      out->backward_ = backward;
    }
    //DEBUG_PRINT("RETURN MULTI\n");
    return out;
  }


  std::shared_ptr<Value<T>> pow(T other) {
    // assert types
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this()};
    auto val = (T)std::pow(data,other);
    //std::string name = "**" + std::to_string(other);
    auto out = std::make_shared<Value<T>>(val, children, OP::OP_POW, "**");

    if(!is_param){
      std::function<void()> backward = [out,other](){
        DEBUG_PRINT("pow %f -> ",out->prev[0]->grad);
        out->prev[0]->grad += other * static_cast<T>(std::pow(other, (other - static_cast<T>(1.0)))) * out->grad;
        DEBUG_PRINT("%f\n",out->prev[0]->grad);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> exp(){
    auto x = data;
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this()};
    auto val = (T)std::exp(x);
    auto out = std::make_shared<Value<T>>(val, children, OP::OP_EXP, "exp");

    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT("exp %f -> ",out->prev[0]->grad);
        out->prev[0]->grad  += out->data * out->grad;
        DEBUG_PRINT("%f\n",out->prev[0]->grad);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> tanh(){
    auto x = data;
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this()};
    auto val = (std::exp(2. * (x)) - 1.0)/(std::exp(2. * (x)) + 1.0);
    auto out = std::make_shared<Value<T>>(val, children, OP::OP_TANH, "tanh");

    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT("tanh %f -> ",out->prev[0]->grad);
        out->prev[0]->grad  += static_cast<T>(1.0 - std::pow(out->data, 2.)) * out->grad;
        DEBUG_PRINT("%f\n",out->prev[0]->grad);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> relu(){
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this()};
    T val = ((this->data) < static_cast<T>(0)) ? static_cast<T>(0.0) : this->data;
    auto out = std::make_shared<Value<T>>(val, children, OP::OP_RELU, "ReLU");
    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT("ReLU %f -> ",out->prev[0]->grad);
        out->prev[0]->grad += static_cast<T>(out->data > static_cast<T>(0)) * out->grad;
        DEBUG_PRINT("%f\n",out->prev[0]->grad);
      };
      out->backward_ = backward;
    }
    return out;
  }

  void backward() {
    std::vector<std::shared_ptr<Value<T>>> topo;
    std::set<std::shared_ptr<Value<T>>> visited;
    std::function<void(std::shared_ptr<Value<T>>)> build_topo = [&](std::shared_ptr<Value<T>> v) {
      if (visited.find(v) == visited.end()) {
        visited.insert(v);
        for (auto child : v->prev) {
          build_topo(child);
        }
        topo.push_back(v);
        std::cout << "Added node " << *v <<std::endl;
      }
    };
    build_topo(this->shared_from_this());
    //std::reverse(topo.begin(), topo.end());

    grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      DEBUG_PRINT("Running for (%p): %f -> ",(void*)it->get(),(*it)->grad);
      (*it)->backward_();
      DEBUG_PRINT("%f\n",(*it)->grad);
    }
  }

  template<typename C>
  friend std::ostream& operator<<(std::ostream& ostr, const Value<C>& m);

};

template<typename T>
std::shared_ptr<Value<T>> operator+(const std::shared_ptr<Value<T>>& lhs,
                                    const std::shared_ptr<Value<T>>& rhs){
  return lhs->add(rhs);
}
template<typename T>
std::shared_ptr<Value<T>> operator*(const std::shared_ptr<Value<T>>& lhs,
                                    const std::shared_ptr<Value<T>>& rhs){
  return lhs->mmul(rhs);
}
template<typename T>
std::shared_ptr<Value<T>> operator*(const std::shared_ptr<Value<T>>& lhs,
                                    const T& rhs){
  auto val = std::make_shared<Value<T>>(rhs);
  return lhs->mmul(val);
}
template<typename T>
std::shared_ptr<Value<T>> operator/(const std::shared_ptr<Value<T>>& lhs,
                                    const std::shared_ptr<Value<T>>& rhs){
  return lhs->div(rhs);
}
template<typename T>
std::shared_ptr<Value<T>> operator-(const std::shared_ptr<Value<T>>& lhs,
                                    const std::shared_ptr<Value<T>>& rhs){
  return lhs->sub(rhs);
}


template<typename T>
std::ostream& operator<<(std::ostream& ostr, const Value<T>& val){
  std::ostringstream stringStream;
  if(val.prev.size() > 0){
    if(val.prev.size() >1){
      stringStream << "[" << val.prev[0] <<" | " <<val.prev[1] << "]";
    } else {
      stringStream << "[" << val.prev[0] << "]";
    }
  }
  std::string childStr = stringStream.str();
  
  DEBUG_PRINT(
    "Value[%p](data=%f, label=%s, op=%d, grad=%f, children=%s)",
    (void*)&val,
    val.data,
    (val.label == nullptr ? "" : val.label),
    val.op,
    val.grad,
    childStr.c_str()
  );
  return ostr;
}
