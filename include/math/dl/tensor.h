#pragma once

#define MEM_ALIGN 16
#define MAX_DIMS 5
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

struct tensor {
  enum datatypes dtype;

  int dimensions;
  int number_elements[MAX_DIMS];
  int nb[MAX_DIMS];

  void *data;
  char *pad[4];
};

struct cgraph {
  int n_nodes;
  int n_leafs;

  struct tensor * nodes[MAX_NODES];

};

struct init_params {
  size_t mem_size;
  void* mem_buffer;
};


struct context * tensor_init(struct init_params params);
void tensor_free(struct context * ctx);

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
#ifndef __cplusplus
} // extern C
#endif
