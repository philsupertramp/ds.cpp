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

enum task_type {
  TASK_INIT = 0,
  TASK_COMPUTE,
  TASK_FINALIZE,
};

struct ComputeParams {
  enum task_type type;

  int ith, nth;

  // worker buffer
  size_t wsize;
  void *wdata;
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
  friend CGraph;
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
  friend void compute_forward_add_f32(const ComputeParams*, const Tensor*, const Tensor*, Tensor*);
  friend void compute_forward_add(const ComputeParams*, const Tensor*, const Tensor*, Tensor*);
  friend void compute_forward(const ComputeParams*, Tensor*);
  friend void compute_backward(Tensor*, bool);
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
  friend void compute_forward(ComputeParams*, Tensor*);
};

