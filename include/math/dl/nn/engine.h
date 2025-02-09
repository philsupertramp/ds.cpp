#pragma once

#include <functional>
#include <memory>
#include <math.h>
#include <vector>
#include <cstdio>

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
{
public:
  T* data;
  std::vector<std::shared_ptr<Value<T>>> prev;
  OP op;
  const char* label;
  float grad;

  std::function<void(const std::shared_ptr<Value<T>>&)> backward_;
  
  Value(const T data_)
  {
    data = new T;
    (*data) = data_;
    prev = {};
    op = OP::OP_NONE;
    label = "";
    grad = 0.0f;
    backward_ = [](const std::shared_ptr<Value<T>>&){};
  }
  Value(const T* data)
  : data(data)
  {
    prev = {};
    op = OP::OP_NONE;
    label = "";
    grad = 0.0f;
    backward_ = [](const std::shared_ptr<Value<T>>&){};
  }

  Value(T* data)
  : data(data)
  {
    prev = {};
    op = OP::OP_NONE;
    label = "";
    grad = 0.0f;
    backward_ = [](const std::shared_ptr<Value<T>>&){};
  }

  Value(T* data, std::vector<std::shared_ptr<Value<T>>> children, OP op, const char* label)
  : data(data), prev(children), op(op), label(std::move(label)), grad(0.0f)
  {
    backward_ = [](const std::shared_ptr<Value<T>>&){};
    DEBUG_PRINT("Constructor, NEW OBJECT!\n");
  }
  Value(const Value& other)
  : data(new T(*other.data)),
    prev(other.prev),
    op(other.op),
    label(other.label),grad(other.grad),backward_(other.backward_)
  {
    if(other.prev.size() == 2){
    DEBUG_PRINT("const Value COPY CONSTRUCTOR %f | %f\n",other.prev[0]->grad,other.prev[1]->grad);
    } else if (other.prev.size() == 1){
    DEBUG_PRINT("const Value COPY CONSTRUCTOR %f\n",other.prev[0]->grad);
    }
  }
  Value(Value* other)
  : data(new T(*other->data)),
    prev(other->prev),
    op(other->op),
    label(other->label),
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
  : data(other.data),
    prev(other.prev),
    op(other.op),
    label(other.label),
    grad(other.grad),
    backward_(std::move(other.backward_)) {DEBUG_PRINT("MOVE\n");}

  ~Value(){
    /*
    if(data != nullptr){
      DEBUG_PRINT("DELETE MFS\n");
      delete data;
    }
    */
  }

  Value& operator=(Value&& other) noexcept {
    DEBUG_PRINT("ASSIGN\n");
    if (this != &other) {
      data = std::move(other.data);
      prev = other.prev;
      op = other.op;
      label = other.label;
      grad = other.grad;
      backward_ = std::move(other.backward_);
    }
      return *this;
  }

  bool operator==(const Value<T>& rhs){
    return *this->data == *rhs.data;
  }

  // OPERATORS
  //
  Value<T> operator+(Value<T>& other){
    std::vector<std::shared_ptr<Value<T>>> children = {std::make_shared<Value<T>>(*this), std::make_shared<Value<T>>(other)};
    auto val = std::make_unique<T>(*(T*)data + (*(T*)other.data));
    auto out = Value<T>(val.release(), children, OP::OP_ADD, "+");

    std::function<void(std::shared_ptr<Value<T>>)> backward = [](const std::shared_ptr<Value<T>>& t){
      DEBUG_PRINT("add\n");
      std::shared_ptr<Value<T>> this_ = t->prev[0];
      std::shared_ptr<Value<T>> other = t->prev[1];
      (*this_).grad += static_cast<T>(1.0) * t->grad;
      other->grad += 1.0 * t->grad;
    };
    out.backward_ = backward;
    return out;
  }

  Value<T> operator*(Value<T>& other) {
    std::vector<std::shared_ptr<Value<T>>> children = {std::make_shared<Value<T>>(*this), std::make_shared<Value<T>>(other)};
    DEBUG_PRINT("MULTI PAST CHILDREN\n");
    auto val = std::make_unique<T>(*(T*)data * (*(T*)other.data));
    Value<T> out(val.release(), children, OP::OP_MUL, "*");
    DEBUG_PRINT("MULTI PAST CONSTRUCTOR\n");
    std::function<void(std::shared_ptr<Value<T>>)> backward = [](const std::shared_ptr<Value<T>>& t){
      auto this_ = t->prev[0];
      auto other = t->prev[1];
      DEBUG_PRINT("mul %f(%p) * %f\n",this_->grad, this_,other->grad);
      (*this_).grad += *(T*)t->data * other->grad;
      (*other).grad += ((*(T*) this_->data) * t->grad);
      DEBUG_PRINT("mul %f(%p) * %f after\n",this_->grad, this_, other->grad);
    };
    out.backward_ = backward;
    DEBUG_PRINT("RETURN MULTI\n");
    return out;
  }
  Value<T>operator*(T other) { return Value<T>(other) * (*this);}

  Value<T> operator-(Value<T>& other){ Value<T> neg(-1.0); Value<T> out = (neg * other) + *this; out.op = OP::OP_SUB; return out; }

  Value<T> operator+=(Value<T>& other){ *this = other + (*this); return *this; }
  Value<T> operator-=(Value<T>& other){ *this = *this - other; return *this; }
  Value<T> operator*=(Value<T>& other){ *this = *this * other; return *this;}
  Value<T> operator/=(Value<T>& other){ *this = *this / other; return *this;}

  Value<T> operator/(Value<T>& other){ auto res = other.pow(-1.) * (*this); res.op = OP::OP_DIV; return res; }

  Value<T> pow(T other) {
    // assert types
    std::vector<std::shared_ptr<Value<T>>> children = {std::make_shared<Value<T>>(*this)};
    auto val = std::make_unique<T>(std::pow(*(T*)(data), other));
    std::string name = "**" + std::to_string(other);
    auto out = Value<T>(val.release(), children, OP::OP_POW, name.c_str());

    std::function<void(std::shared_ptr<Value<T>>)> backward = [other](const std::shared_ptr<Value<T>>& t){
      DEBUG_PRINT("pow\n");
      std::shared_ptr<Value<T>> this_ = t->prev[0];
      (*this_).grad += other * std::pow(other, (other - (T)1.0)) * t->grad;
    };
    out.backward_ = backward;
    return out;
  }

  Value<T> exp(){
    auto x = data;
    std::vector<std::shared_ptr<Value<T>>> children = {std::make_shared<Value<T>>(*this)};
    auto val = std::make_unique<T>((T)std::exp(*x));
    auto out = Value(val.release(), children, OP::OP_EXP, "exp");

    std::function<void(std::shared_ptr<Value<T>>)> backward = [](const std::shared_ptr<Value<T>>& t){
      DEBUG_PRINT("exp\n");
      std::shared_ptr<Value<T>> this_ = t->prev[0];
      (*this_).grad  += *(T*)t->data * t->grad;
    };
    out.backward_ = backward;
    return out;
  }

  Value<T> tanh(){
    auto x = data;
    std::vector<std::shared_ptr<Value<T>>> children = {std::make_shared<Value<T>>(*this)};
    auto val = std::make_unique<T>(
      (std::exp(2. * (*x)) - 1.0)/(std::exp(2. * (*x)) + 1.0) 
    );
    auto out = Value(val.release(), children, OP::OP_TANH, "tanh");

    std::function<void(std::shared_ptr<Value<T>>)> backward = [](const std::shared_ptr<Value<T>>& t){
      DEBUG_PRINT("tanh\n");
      std::shared_ptr<Value<T>> this_ = t->prev[0];
      T tval = *(T*)t->data;
      (*this_).grad  += (1.0 - std::pow(tval, 2.)) * t->grad;
    };
    out.backward_ = backward;
    return out;
  }

  Value<T> relu(){
    std::vector<std::shared_ptr<Value<T>>> children = {std::make_shared<Value<T>>(*this)};
    auto val = std::make_unique<T>(
        ((*(T*)this->data) < 0) ? static_cast<T>(0.0) : *this->data
    );
    Value<T> out(val.release(),children,OP::OP_RELU,"ReLU");
    std::function<void(std::shared_ptr<Value<T>>)> backward = [](const std::shared_ptr<Value<T>>& t){
      DEBUG_PRINT("ReLU\n");
      std::shared_ptr<Value<T>> this_ = t->prev[0];
      (*this_).grad += static_cast<T>(*(T*)t->data > static_cast<T>(0)) * t->grad;
    };
    out.backward_ = backward;
    return out;
  }

  void _build_topo(const std::shared_ptr<Value<T>>& v, std::vector<std::shared_ptr<Value<T>>>&topo,std::vector<std::shared_ptr<Value<T>>>&visited){
    if(v != nullptr && std::find(visited.begin(), visited.end(), v) == visited.end()){
      visited.push_back(v);
      for(size_t i = 0; i < v->prev.size(); i++){
        _build_topo(v->prev[i],topo,visited);
      }
      topo.push_back(v);
    }
  }

  void backward(){
    std::vector<std::shared_ptr<Value<T>>> topo = {};
    std::vector<std::shared_ptr<Value<T>>> visited = {};

    this->grad = static_cast<T>(1.0);
    DEBUG_PRINT("CREATING SHARED PTR of this\n");
    std::cout <<*this<<std::endl;
    std::shared_ptr<Value<T>> _this = std::make_shared<Value<T>>(*this);

    DEBUG_PRINT("BUILD TOPOLOGY %f[%p]\n",_this->grad,_this);
    _build_topo(_this,topo,visited);

    DEBUG_PRINT("compute gradients\n");
    std::reverse(topo.begin(),topo.end());
    for(auto node : topo){
      if(node != nullptr){
        node->backward_(node);
      }
    }
    *this = _this.get();
    std::cout <<*this<<std::endl;
  }

  template<typename C>
  friend std::ostream& operator<<(std::ostream& ostr, const Value<C>& m);
};

template<typename T>
std::ostream& operator<<(std::ostream& ostr, const Value<T>& val){
  ostr << "Value(data=" << *(T*)(val.data) << ", label=" << val.label << ", op=" << val.op << ", grad=" << val.grad << ")";
  return ostr;
}
