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

inline void vec_add_f32(const int ne, float *z, float *x, float *y){for(int i = 0; i < ne; ++i) z[i] = x[i] + y[i]; }

void compute_forward_add_f32(const ComputeParams* params, const Tensor* srcL, const Tensor* srcR, Tensor* dst){
  //assert same shapes
  //
  const int ith = params->ith;
  const int nth = params->nth;

  const int n = srcL->nrows();
  const int nc = srcL->num_elem_[0];

  const size_t nb00 = srcL->num_bytes_[0];
  const size_t nb01 = srcL->num_bytes_[1];
  
  const size_t nb10 = srcR->num_bytes_[0];
  const size_t nb11 = srcR->num_bytes_[1];

  const size_t nb0 = dst->num_bytes_[0];
  const size_t nb1 = dst->num_bytes_[1];

  assert(nb0 == sizeof(float));
  assert(nb00 == sizeof(float));

  if(nb10 == sizeof(float)){
    const int j0 = (n/nth)*ith;
    const int j1 = ith == nth - 1 ? n : (n/nth) * (ith + 1);

    for(int j = j0; j < j1; j++){
      vec_add_f32(
        nc,
        (float*) ((char*) dst->data_ + j * nb1),
        (float*) ((char*) srcL->data_ + j * nb01),
        (float*) ((char*) srcR->data_ + j * nb11)
      );
    }
  } else {
    for(int j = ith; j < n; j += nth){
      float *dst_ptr = (float*) ((char*) dst->data_ + j * nb1);
      float *srcL_ptr = (float*) ((char*) srcL->data_ + j * nb1);
      for(int i = 0; i < nc; i++){
        float * srcR_ptr = (float*) ((char*) srcR->data_ + j * nb11 + i * nb10);
        dst_ptr[i] = srcL_ptr[i] + *srcR_ptr;
      }
    }
  }
}

void compute_forward_add(const ComputeParams* params, const Tensor* srcL, const Tensor* srcR, Tensor* dst){
  switch(srcL->dtype_){
    case datatypes::TYPE_FLOAT:
      {
        compute_forward_add_f32(params, srcL, srcR, dst);
      }
      break;
    case datatypes::TYPE_COUNT:
    default:
      {}
      break;
  }
}

void compute_forward(const ComputeParams* params, Tensor* t){
  switch(t->op_){
    case ops::OP_ADD:
      {
        compute_forward_add(params, t->srcL_, t->srcR_, t);
      }
      break;
    case ops::OP_NONE:
    default:
      {
        /// should not happen
        assert(false);
      }
      break;
  }
}

void compute_backward(Tensor* t, bool inplace){
  Tensor* left = t->srcL_;
  Tensor* right = t->srcR_;

  switch(t->op_){
    case ops::OP_ADD:
      {
        if(left->grad_){
          left->grad_ = add(left->grad_, t->grad_, inplace);
        }
        if(right->grad_){
          right->grad_ = add(right->grad_, t->grad_, inplace);
        }
      }
      break;
    case ops::OP_NONE:
    default:
      {
        // this should not happen
        assert(false);
      }
      break;
  }

  
}

void visit_parents(CGraph* graph, Tensor* node){
  if(node->grad_ == NULL){
    if(node->op_ != OP_NONE){}
  }

  for(size_t i = 0; i < graph->n_nodes; i++){
    if(graph->nodes[i] == node){
      return;
    }
  }
  for(size_t i = 0; i < graph->n_leafs; i++){
    if(graph->nodes[i] == node){
      return;
    }
  }

  if(node->srcL_){
    visit_parents(graph, node->srcL_);
  }
  if(node->srcR_){
    visit_parents(graph, node->srcR_);
  }
}

void build_forward_impl(CGraph* graph, Tensor* t, bool expand){
  if(!expand){
    graph->n_nodes = 0;
    graph->n_leafs = 0;
  }

  const size_t n0 = graph->n_nodes;

  visit_parents(graph, t);

  const size_t n_new = graph->n_nodes - n0;

  if(n_new > 0){
    assert(graph->nodes[graph->n_nodes - 1] == t);
  }
}

CGraph build_forward(Tensor* t){
  CGraph res;
  build_forward_impl(&res, t, false);
  return res;
}

CGraph build_forward_expand(Tensor *t){
  CGraph res;
  build_forward_impl(&res, t, true);
  return res;
}

CGraph build_backward(CGraph* graph, bool keep=true){
  CGraph res = *graph;

  assert(graph->n_nodes > 0);

  if(keep){
    for(size_t i = 0; i < graph->n_nodes; i++){
      Tensor* t = graph->nodes[i];

      if(t->grad_){
        t->grad_ = t;
        graph->grads[i] = t->grad_;
      }
    }
  }

  for(int i = graph->n_nodes - 1; i >= 0; i--){
    Tensor* t = graph->nodes[i];
    if(t->grad_){
      compute_backward(t, keep);
    }
  }
  for(int i = graph->n_nodes - 1; i >= 0; i--){
    Tensor* t = graph->nodes[i];
    if(t->is_param_){
      build_forward_impl(&res, t->grad_, true);
    }
  }

  return res;
}

