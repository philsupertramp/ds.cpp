#include "../../include/math/dl/tensor.h"


Tensor* add(Tensor* a, Tensor* b, bool inplace){
  bool is_node = false;

  if(!inplace && (a->hasGrad() || b->hasGrad())){
    is_node = true;
  }

  Tensor *result = inplace ? a->view() : a->dup();
  result->op_ = ops::OP_ADD;
  result->grad_ = is_node ? result->dup() : NULL;
  result->srcL_ = a;
  result->srcR_ = b;

  return result;
}

Tensor* add(Tensor* a, Tensor* b){
  return add(a, b, false);
}
