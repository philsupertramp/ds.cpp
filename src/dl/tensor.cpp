#include "../../include/math/dl/tensor.h"
#include <stdexcept>

const size_t TYPE_SIZE[TYPE_COUNT] = {
    sizeof(int),
    sizeof(float),
    sizeof(double),
};

struct object {
  size_t offset;
  size_t size;

  struct object * next;
  enum datatypes type;

  char padding[4];
};

const size_t OBJECT_SIZE = sizeof(struct object);

struct context {
  size_t mem_size;
  void * mem_buffer;
  bool    mem_buffer_owned;
  bool    no_alloc;
  int     n_objects;

  struct object * begin;
  struct object * end;
};

struct context_container {
  bool used;

  struct context context;
};

struct state {
  struct context_container contexts[MAX_CONTEXTS];
};

struct state global_state;


struct context * tensor_init(struct init_params params){
  struct context * ctx = NULL;

  static bool cold_start = true;
  if(cold_start){
    for(int i = 0; i < MAX_CONTEXTS; ++i){
      global_state.contexts[i].used = false;
    }
    cold_start = false;
  }

  for(int i = 0; i < MAX_CONTEXTS; ++i){
    if(!global_state.contexts[i].used){
      global_state.contexts[i].used = true;
      ctx = &global_state.contexts[i].context;

      break; // found unused context, continue
    }
  }

  if(ctx == NULL){
    // didn't find context, all context' used.
    return NULL;
  }

  *ctx = (struct context){
    .mem_size         = params.mem_size,
    .mem_buffer       = params.mem_buffer ? params.mem_buffer : malloc(params.mem_size),
    .mem_buffer_owned = params.mem_buffer ? false : true,
    .n_objects        = 0,
    .begin            = NULL,
    .end              = NULL,
  };

  // assert_aligned(ctx->mem_buffer);
  return ctx;
}
void tensor_free(struct context * ctx){
  for(int i = 0; i < MAX_CONTEXTS; ++i){
    if(&global_state.contexts[i].context == ctx) {
      global_state.contexts[i].used = false;

      if(ctx->mem_buffer_owned){
        free(ctx->mem_buffer);
      }
    }
  }
}

static struct tensor * new_tensor_impl(
  struct context      * ctx,      // context
  enum   datatypes      type,     //datatype of elements
  int                   n_dims,   // number of dimensions
  const int       * ne,       // vector of numbers of elements
  void* data) {
  struct object * current_object = ctx->end;

  const size_t current_offset = current_object == NULL ? 0 : current_object->offset;
  const size_t current_size = current_object == NULL ? 0 : current_object->size;
  const size_t current_end = current_offset + current_size;

  size_t required_size = 0;

  if(data == NULL){
    required_size += TYPE_SIZE[type];
    for(int i = 0; i < n_dims; ++i){
      required_size *= ne[i];
    }

    required_size = ((required_size + MEM_ALIGN - 1)/MEM_ALIGN)*MEM_ALIGN;
  }

  required_size += sizeof(struct tensor);

  if(current_end + required_size + OBJECT_SIZE > ctx->mem_size){
    throw std::invalid_argument("Not enough space in context's memory pool");
  }

  char * const mem_buffer = (char*) ctx->mem_buffer;

  struct object * const obj_new = (struct object*)(mem_buffer + current_end);

  *obj_new = (struct object){
    .offset = current_end + OBJECT_SIZE,
    .size = required_size,
    .next = NULL,
  };

  if(current_object != NULL){
    current_object->next = obj_new;
  } else {
    ctx->begin = obj_new;
  }

  ctx->end = obj_new;

  struct tensor * const result = (struct tensor*)(mem_buffer + obj_new->offset);

  *result = (struct tensor){
    .dtype = type,
    .dimensions=n_dims,
    .number_elements={1, 1, 1, 1},
    .nb={0, 0, 0, 0},
    //.op=OP_NONE,
    //.is_param=false,
    //.grad=NULL,
    //.src0=NULL,
    //.src1=NULL,
    //.n_tasks=0,
    //.perf_runs=0,
    //.perf_cycles=0,
    //.perf_time_us=0,
    .data=data==NULL?(void *)(result + 1) : data,
    .pad={ 0 },
  };

  for(int i = 0; i < n_dims; ++i){
    result->number_elements[i] = ne[i];
  }
  result->nb[0] = TYPE_SIZE[type];
  for(int i = 1; i < MAX_DIMS; ++i){
    result->nb[i] = result->nb[i - 1] * result->number_elements[i - 1];
  }
  return result;
}


struct tensor* new_tensor(
  struct context * ctx,
  enum datatypes type,
  int n_dims,
  const int *ne){
  return new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

struct tensor* new_tensor_1d(
  struct context * ctx,
  enum datatypes type,
  int ne0) {
  return new_tensor(ctx, type, 1, &ne0);
}

struct tensor* new_tensor_2d(
  struct context * ctx,
  enum datatypes type,
  int ne0,
  int ne1) {
  const int ne[2] = {ne0, ne1};
  return new_tensor(ctx, type, 2, ne);
}
struct tensor* new_tensor_3d(
  struct context * ctx,
  enum datatypes type,
  int ne0,
  int ne1,
  int ne2) {
  const int ne[3] = {ne0, ne1, ne2};
  return new_tensor(ctx, type, 3, ne);
}
struct tensor* new_tensor_4d(
  struct context * ctx,
  enum datatypes type,
  int ne0,
  int ne1,
  int ne2,
  int ne3) {
  const int ne[4] = {ne0, ne1, ne2, ne3};
  return new_tensor(ctx, type, 4, ne);
}



