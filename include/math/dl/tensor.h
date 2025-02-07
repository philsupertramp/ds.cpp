#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>


#define MAX_DIMS 4
#define MAX_NODES 1024
#define MEM_ALIGN 16


enum datatypes {
  TYPE_FLOAT = 0,
  TYPE_INT,
  TYPE_COUNT
};

enum ops {
  OP_ADD,

  OP_NONE,
};

const size_t TYPE_SIZE[TYPE_COUNT] = {
  sizeof(float),
  sizeof(int),
};


class CGraph;


class Tensor
{
private:
  enum datatypes dtype_;
  int dims_;

  size_t num_elem_[MAX_DIMS];
  size_t num_bytes_[MAX_DIMS];

  enum ops op_;

  bool is_param_;

  Tensor *grad_;
  Tensor *srcL_;
  Tensor *srcR_;

  void *data_;
  char *pad_[MAX_DIMS];

  bool used_ = false;

public:
  Tensor()
  {}

  Tensor(size_t dims, const size_t *num_elements, void* data, enum datatypes type)
  : dtype_(type), dims_(dims)
  {
    size_t required_size = 0;
    if(data == NULL){
      required_size += TYPE_SIZE[type];
      for(size_t i = 0; i < dims; ++i){
        required_size *= num_elements[i];
      }

      required_size = ((required_size + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
      required_size += sizeof(Tensor);
    }

    for (size_t i = 0; i < MAX_DIMS; ++i) {
      num_elem_[i] = i < dims ? num_elements[i] : 1;
    }
    num_bytes_[0] = TYPE_SIZE[type];
    for(size_t i = 1; i <= MAX_DIMS; ++i){
      num_bytes_[i] = num_bytes_[i - 1] * num_elem_[i - 1];
    }

    data_ = (data == NULL ? (void *)(malloc(required_size)) : data);
    grad_ = NULL;
    srcL_ = NULL;
    srcR_ = NULL;
    op_ = ops::OP_NONE;
    is_param_ = false;
    used_ = (data == NULL);
    pad_[0] = 0;
  }

  ~Tensor(){
    if(used_){
      free(data_);
      used_ = false;
    }
  }

  [[nodiscard]] inline size_t nrows() const {
    return num_elem_[1] * num_elem_[2] * num_elem_[3];
  }

  [[nodiscard]] inline bool hasGrad() const { return this->grad_ != NULL; }

  // attribute getters
  Tensor * srcL(){ return srcL_; }
  Tensor * srcR(){ return srcR_; }
  Tensor * grad(){ return grad_; }


  Tensor* view(){
    auto t = new Tensor(dims_, num_elem_, data_, dtype_);
    return t;
  }

  Tensor* dup(){
    auto t = new Tensor(dims_, num_elem_, NULL, dtype_);
    return t;
  }

  friend Tensor* add(Tensor*, Tensor*, bool);
  friend Tensor* add(Tensor*, Tensor*);

  friend void visit_parents(CGraph*, Tensor *);
  friend CGraph build_backward(CGraph* graph, bool keep);
};

class CGraph
{
  size_t n_nodes;
  size_t n_leafs;

  Tensor* nodes[MAX_NODES];
  Tensor* grads[MAX_NODES];
  Tensor* leafs[MAX_NODES];
public:
  CGraph() = default;


  friend void build_forward_impl(CGraph*, Tensor *, bool);
  friend void visit_parents(CGraph*, Tensor*);
  friend CGraph build_backward(CGraph* graph, bool keep);
  friend void compute_backward(Tensor*, bool);
};
void compute_backward(Tensor* t, bool keep){

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

