#pragma once

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <functional>
#include <vector>
#include <algorithm>
#include "../../format.h"

#define DEBUG 1

#if (DEBUG < 1)
#define RESET ""
#define BLACK ""
#define RED ""
#define GREEN ""
#define YELLOW      ""
#define BLUE        ""
#define MAGENTA     ""
#define CYAN        ""
#define WHITE       ""
#define BOLDBLACK   ""
#define BOLDRED     ""
#define BOLDGREEN   ""
#define BOLDYELLOW  ""
#define BOLDBLUE    ""
#define BOLDMAGENTA ""
#define BOLDCYAN    ""
#define BOLDWHITE   ""
#define DEBUG_PRINT(...)
#define DEBUG_PRINT_1(...)
#define DEBUG_PRINT_2(...)
#else
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#endif

#if (DEBUG == 1)
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#define DEBUG_PRINT_1(...)
#define DEBUG_PRINT_2(...)
#elif (DEBUG == 2)
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#define DEBUG_PRINT_1(...) printf(__VA_ARGS__)
#define DEBUG_PRINT_2(...)
#elif (DEBUG == 3)
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#define DEBUG_PRINT_1(...) printf(__VA_ARGS__)
#define DEBUG_PRINT_2(...) printf(__VA_ARGS__)
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


template<typename T>
class Value
: public std::enable_shared_from_this<Value<T>>
{
public:
  T data;
  T grad;

  std::shared_ptr<Value<T>> left = nullptr;
  std::shared_ptr<Value<T>> right = nullptr;
  OP op;
  
  std::string label;

  std::function<void()> backward_;
  bool is_param = false;
  
  Value(const T data_, bool is_param_=false)
  : std::enable_shared_from_this<Value<T>>()
  {
    data = data_;
    op = OP::OP_NONE;
    label = "";
    grad = static_cast<T>(0.0f);
    backward_ = [](){};
    is_param = is_param_;
    left = nullptr;
    right = nullptr;

    DEBUG_PRINT_1("%sValue<T> created %s\n", BOLDCYAN, RESET);
  }

  Value(const T data_, OP op_, const std::string& label_, const std::shared_ptr<Value<T>>& left_ = nullptr, const std::shared_ptr<Value<T>>& right_ = nullptr)
  : std::enable_shared_from_this<Value<T>>()
  {
    data = data_;
    op = op_;
    label = label_;
    grad = static_cast<T>(0.0);
    backward_ = [](){};
    left = left_;
    right = right_;
    DEBUG_PRINT_1("%sValue<T> created %s\n", BOLDCYAN, RESET);
  }
  Value(const Value& other)
  : std::enable_shared_from_this<Value<T>>(),
    data(other.data),
    left(other.left),
    right(other.right),
    op(other.op),
    label(other.label),grad(other.grad),backward_(std::move(other.backward_))
  {
    if(other.left != nullptr){
      DEBUG_PRINT("%sconst Value COPY CONSTRUCTOR [%p] %f", BOLDMAGENTA, (void*)this, this->left->grad);
      if(other.right != nullptr){
        DEBUG_PRINT(" | %f", this->right->grard);
      }
      DEBUG_PRINT("%s", RESET);
    }
  }

  Value(Value* other)
  : std::enable_shared_from_this<Value<T>>(),
    data(other->data),
    left(other->left),
    right(other->right),
    op(other->op),
    label(other->label),
    grad(other->grad),
    backward_(std::move(other->backward_))
  {
    if(other.left != nullptr){
      DEBUG_PRINT("%sValue COPY CONSTRUCTOR [%p] %f", BOLDMAGENTA, (void*)this, this->left->grad);
      if(other.right != nullptr){
        DEBUG_PRINT(" | %f", this->right->grard);
      }
      DEBUG_PRINT("%s", RESET);
    }
  }
  Value(Value&& other) noexcept
  : std::enable_shared_from_this<Value<T>>(),
    data(other.data),
    left(other->left),
    right(other->right),
    op(other.op),
    label(other.label),
    grad(other.grad),
    backward_(std::move(other.backward_)) {DEBUG_PRINT("%sMOVE%s\n", BOLDYELLOW, RESET);}

  ~Value() = default;

  // Assignment and comparison

  Value& operator=(Value&& other) noexcept {
    DEBUG_PRINT("%sASSIGN%s\n", BOLDRED, RESET);
    if (this != &other) {
      data = other.data;
      left = other.left;
      right = other.right;
      op = other.op;
      label = other.label;
      grad = other.grad;
      backward_ = std::move(other.backward_);
    }
      return *this;
  }

  bool operator==(const Value<T>& rhs){
    DEBUG_PRINT("%p %p\n", *this->data,*rhs->data);
    return (*this->data) == (*rhs.data) && left == rhs->left && right == rhs->right;
  }

  // Math OPERATORS

  std::shared_ptr<Value<T>> sub(const std::shared_ptr<Value<T>>& other){
    auto val = data - other->data;
    auto out = std::make_shared<Value<T>>(val, OP::OP_SUB, "-", this->shared_from_this(), other);

    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT_1("%ssub %f(%p) * %f\n", BLUE, out->left->grad, (void*)out->left.get(),out->right->grad);
        out->left->grad += static_cast<T>(1.0) * out->grad;
        out->right->grad += static_cast<T>(-1.0) * out->grad;
        DEBUG_PRINT_1("sub %f(%p) * %f after%s\n",out->left->grad, (void*)out->left.get(),out->right->grad, RESET);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> add(const std::shared_ptr<Value<T>>& other){
    auto val = data + other->data;
    auto out = std::make_shared<Value<T>>(val, OP::OP_ADD, "+", this->shared_from_this(), other);

    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT_1("%sadd %f(%p) * %f\n", BLUE, out->left->grad, (void*)out->left.get(),out->right->grad);
        out->left->grad += static_cast<T>(1.0) * out->grad;
        out->right->grad += static_cast<T>(1.0) * out->grad;
        DEBUG_PRINT_1("add %f(%p) * %f after%s\n",out->left->grad, (void*)out->left.get(),out->right->grad, RESET);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> mul(const std::shared_ptr<Value<T>>& other){
    //DEBUG_PRINT("MULTI PAST CHILDREN\n");
    auto val = data * other->data;
    auto out = std::make_shared<Value<T>>(val, OP::OP_MUL, "*", this->shared_from_this(), other);
    //DEBUG_PRINT("MULTI PAST CONSTRUCTOR\n");
    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT_1("%smul %f(%p) %f(%p) * %f\n", BLUE, out->grad,(void*)out.get(),out->left->grad, (void*)out->left.get(),out->right->grad);
        out->left->grad += out->right->data * out->grad;
        out->right->grad += out->left->data * out->grad;
        DEBUG_PRINT_1("mul %f(%p) * %f after%s\n",out->left->grad, (void*)out->left.get(),out->right->grad, RESET);
      };
      out->backward_ = backward;
    }
    //DEBUG_PRINT("RETURN MULTI\n");
    return out;
  }
  std::shared_ptr<Value<T>> div(const std::shared_ptr<Value<T>>& other){
    //DEBUG_PRINT("MULTI PAST CHILDREN\n");
    auto val = data / other->data;
    auto out = std::make_shared<Value<T>>(val, OP::OP_DIV, "/", this->shared_from_this(), other);
    //DEBUG_PRINT("MULTI PAST CONSTRUCTOR\n");
    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT_1("%sdiv %f(%p) %f(%p) * %f\n", BLUE, out->grad,(void*)out.get(),out->left->grad, (void*)out->left.get(),out->right->grad);
        out->left->grad += (static_cast<T>(1.)/out->right->data) * out->grad;
        out->right->grad += (-out->left->data / (out->right->data * out->right->data)) * out->grad;
        DEBUG_PRINT_1("div %f(%p) * %f after%s\n",out->left->grad, (void*)out->left.get(),out->right->grad, RESET);
      };
      out->backward_ = backward;
    }
    //DEBUG_PRINT("RETURN MULTI\n");
    return out;
  }

  std::shared_ptr<Value<T>> pow(T other) {
    // assert types
    auto val = (T)std::pow(data, other);
    std::string name = ("**" + std::to_string(other));
    auto out = std::make_shared<Value<T>>(val, OP::OP_POW, "**", this->shared_from_this());

    if(!is_param){
      std::function<void()> backward = [out,other](){
        DEBUG_PRINT_1("%spow %f -> ", BLUE, out->left->grad);
        out->left->grad += other * std::pow(out->left->data, (other - static_cast<T>(1.0))) * out->grad;
        DEBUG_PRINT_1("%f%s\n", out->left->grad, RESET);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> exp(){
    auto x = data;
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this()};
    auto val = (T)std::exp(x);
    auto out = std::make_shared<Value<T>>(val, OP::OP_EXP, "exp", this->shared_from_this());

    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT_1("%sexp %f -> ", BLUE, out->left->grad);
        out->left->grad  += out->data * out->grad;
        DEBUG_PRINT_1("%f%s\n", out->left->grad, RESET);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> tanh(){
    auto x = data;
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this()};
    auto val = std::tanh(x);
    auto out = std::make_shared<Value<T>>(val, OP::OP_TANH, "tanh", this->shared_from_this());

    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT_1("%stanh %f -> ", BLUE, out->left->grad);
        out->left->grad  += static_cast<T>(1.0 - std::pow(out->data, 2.)) * out->grad;
        DEBUG_PRINT_1("%f%s\n", out->left->grad, RESET);
      };
      out->backward_ = backward;
    }
    return out;
  }

  std::shared_ptr<Value<T>> relu(){
    std::vector<std::shared_ptr<Value<T>>> children = {this->shared_from_this()};
    T val = ((this->data) < static_cast<T>(0)) ? static_cast<T>(0.0) : this->data;
    auto out = std::make_shared<Value<T>>(val, OP::OP_RELU, "ReLU", this->shared_from_this());
    if(!is_param){
      std::function<void()> backward = [out](){
        DEBUG_PRINT_1("%sReLU %f -> ", BLUE, out->left->grad);
        out->left->grad += static_cast<T>(out->left->data > static_cast<T>(0)) * out->grad;
        DEBUG_PRINT_1("%f%s\n",out->left->grad, RESET);
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
        v->grad= static_cast<T>(0.0);
        visited.insert(v);
        if(v->left != nullptr) {
          build_topo(v->left);
        }
        if(v->right != nullptr) {
          build_topo(v->right);
        }
        topo.push_back(v);
        std::ostringstream ss;
        ss << *v;
        DEBUG_PRINT_2("Added node %s\n", ss.str().c_str());
      }
    };
    build_topo(this->shared_from_this());

    grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
      DEBUG_PRINT_2("Running for (%p): %f -> ",(void*)it->get(),(*it)->grad);
      (*it)->backward_();
      DEBUG_PRINT_2("%f\n",(*it)->grad);
    }
  }

  static void printGraph(const std::shared_ptr<Value<T>>& node, int depth = 0) {
    for(int i = 0; i < depth; ++i){ std::cout << "\t"; }
    int d = depth + 1;
    std::cout << *node << std::endl;
    if(node->left != nullptr){
      std::cout << "L: ";
      printGraph(node->left, d);
    }
    if(node->right != nullptr){
      std::cout << "R: ";
      printGraph(node->right, d);
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
  return lhs->mul(rhs);
}
template<typename T>
std::shared_ptr<Value<T>> operator*(const T& lhs,
                                    const std::shared_ptr<Value<T>>& rhs){
  auto val = std::make_shared<Value<T>>(lhs);
  return val->mul(rhs);
}
template<typename T>
std::shared_ptr<Value<T>> operator*(const std::shared_ptr<Value<T>>& lhs,
                                    const T& rhs){
  auto val = std::make_shared<Value<T>>(rhs);
  return lhs->mul(val);
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
void  operator+=(std::shared_ptr<Value<T>>& lhs,
                                     const std::shared_ptr<Value<T>>& rhs){
  lhs = lhs + rhs;
}


template<typename T>
std::ostream& operator<<(std::ostream& ostr, const Value<T>& val){
  std::ostringstream stringStream;
  if(val.left != nullptr){
    stringStream << "[" << val.left;
    if(val.right != nullptr){
      stringStream << " | " << val.right;
    }
    stringStream << "]";
  }
  std::string childStr = stringStream.str();
  
  ostr << format(
    "Value[%p](data=%f, label=%s, op=%d, grad=%f, children=%s)",
    (void*)&val,
    val.data,
    val.label.c_str(), 
    val.op,
    val.grad,
    childStr.c_str()
  );
  return ostr;
}
