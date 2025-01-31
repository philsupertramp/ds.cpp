#pragma once

#define MEM_ALIGN 16
#define MAX_DIMS 4
#define MAX_NODES 1024
#define MAX_CONTEXTS 32   // The number of contexts that are allowed to exist in parallel


#include <memory>
#include <cstddef>
#include <cstdint>
#ifndef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

struct objects;
struct context;

enum datatypes {
  INT,
  FLOAT32,
  DOUBLE,
  TYPE_COUNT
};

enum ops {
  OP_NONE = 0,
  // internal
  OP_DUP,

  // math
  OP_ADD,
};

struct tensor {
  enum datatypes dtype;

  int dimensions;
  int number_elements[MAX_DIMS];

  /* Number of bytes
   * number_bytes[0] = sizeof(dtype)
   * number_bytes[1] = number_bytes[0]     * number_elements[0]     + padding
   * number_bytes[i] = number_bytes[i - 1] * number_elements[i - 1]
   */
  int number_bytes[MAX_DIMS];

  enum ops op;

  bool is_param;

  tensor *grad;
  tensor *src0;
  tensor *src1;

  void *data;
  char *pad[4];
};

struct cgraph {
  int n_nodes;
  int n_leafs;

  struct tensor * nodes[MAX_NODES];
  struct tensor * grads[MAX_NODES];
  struct tensor * leafs[MAX_NODES];

};

struct init_params {
  size_t mem_size;
  void* mem_buffer;
};

void print_object(const struct object * obj);
void print_objects(const struct context * ctx);


struct context * tensor_init(struct init_params params);
void tensor_free(struct context * ctx);

// Factory methods

struct tensor* new_tensor(
  struct context * ctx,
  enum datatypes type,
  int n_dims,
  const int *ne);

struct tensor * new_tensor_1d(
        struct context * ctx,
        enum   datatypes type,
        int    ne0);

struct tensor * new_tensor_2d(
        struct context * ctx,
        enum   datatypes type,
        int    ne0,
        int    ne1);

struct tensor * new_tensor_3d(
        struct context * ctx,
        enum   datatypes type,
        int    ne0,
        int    ne1,
        int    ne2);

struct tensor * new_tensor_4d(
        struct context * ctx,
        enum   datatypes type,
        int    ne0,
        int    ne1,
        int    ne2,
        int    ne3);

// Getters
int nrows(
  struct tensor* t);

// Move operators
struct tensor *view_tensor(
  struct context * ctx,
  struct tensor *t);

struct tensor *dup_impl(
  struct context * ctx,
  struct tensor *t,
  bool inplace);
struct tensor *dup(
  struct context * ctx,
  struct tensor *t);
struct tensor *dup_inplace(
  struct context * ctx,
  struct tensor *t);


struct tensor *dup_tensor(
  struct context * ctx,
  struct tensor *t);


struct tensor * set_f32(
  struct tensor *t,
  float val);

float get_f32_1d(
  const struct tensor *t,
  size_t index);



// Math

// Addition
struct tensor * add_impl(
  struct context * ctx,
  struct tensor * a,
  struct tensor * b,
  bool inplace);

struct tensor * add(
  struct context * ctx,
  struct tensor * a,
  struct tensor * b);

struct tensor * add_inplace(
  struct context * ctx,
  struct tensor * a,
  struct tensor * b);


// Compute Graph

void set_param(
  struct context * ctx,
  struct tensor *t);

void build_forward_impl(
  struct cgraph* graph,
  struct tensor* t,
  bool expand);

struct cgraph build_forward(
  struct tensor *t);

void build_forward_expand(
  struct cgraph* graph,
  struct tensor *t);

struct cgraph build_backward(
  struct tensor* t,
  struct cgraph* graph,
  bool keep);

#ifndef __cplusplus
} // extern C
#endif
