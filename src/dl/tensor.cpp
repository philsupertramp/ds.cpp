#include "../../include/math/dl/tensor.h"
#include <cassert>
#include <stdexcept>
#include <cstdio>

#define DL_DEBUG 1

#if (DL_DEBUG >= 1)
#define DL_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define DL_PRINT_DEBUG(...)
#endif

#if (DL_DEBUG >= 5)
#define DL_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define DL_PRINT_DEBUG_5(...)
#endif

#if (DL_DEBUG >= 10)
#define DL_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define DL_PRINT_DEBUG_10(...)
#endif

#define DL_PRINT(...) printf(__VA_ARGS__)


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

// data type conversions
inline static void vec_set_i32(const int n, int *x, const int v){ for (int i = 0; i < n; ++i) x[i] = v; }
inline static void vec_set_f32(const int n, float *x, const float v){ for (int i = 0; i < n; ++i) x[i] = v; }
inline static void vec_set_d32(const int n, double *x, const double v){ for (int i = 0; i < n; ++i) x[i] = v; }



void print_object(const struct object * obj){
  DL_PRINT(" - object: offset = %zu, size = %zu, next = %p\n",
           obj->offset, obj->size, (const void*) obj->next);
}
void print_objects(const struct context * ctx){
  struct object * obj = ctx->begin;

  DL_PRINT("%s: objects in context %p:\n", __func__, (const void*) ctx);

  while (obj != NULL){
    print_object(obj);
    obj = obj->next;
  }

  DL_PRINT("%s: --- end ---\n", __func__);
}

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
    // didn't find context, all contexts used.
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
    .number_bytes={0, 0, 0, 0},
    .op=ops::OP_NONE,
    .is_param=false,
    .grad=NULL,
    .src0=NULL,
    .src1=NULL,
    //.n_tasks=0,
    //.perf_runs=0,
    //.perf_cycles=0,
    //.perf_time_us=0,
    .data=data==NULL ? (void *)(result + 1) : data,
    .pad={ 0 },
  };

  for(int i = 0; i < n_dims; ++i){
    result->number_elements[i] = ne[i];
  }
  result->number_bytes[0] = TYPE_SIZE[type];
  for(int i = 1; i <= MAX_DIMS; ++i){
    result->number_bytes[i] = result->number_bytes[i - 1] * result->number_elements[i - 1];
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

// Getters
int nrows(
  struct tensor *t){
  return t->number_elements[1] * t->number_elements[2] * t->number_elements[3];
}

// Move
struct tensor *view_tensor(
  struct context * ctx,
  struct tensor *t){
  return new_tensor_impl(ctx, t->dtype, t->dimensions, t->number_elements, t->data);
}



struct tensor *dup_impl(
  struct context * ctx,
  struct tensor *t,
  bool inplace){
  bool is_node = false;

  if(!inplace && t->grad){
    is_node = true;
  }

  struct tensor * result = inplace ? view_tensor(ctx, t) : dup_tensor(ctx, t);

  result->op = ops::OP_DUP;
  result->grad = is_node ? dup_tensor(ctx, result) : NULL;
  result->src0 = t;
  result->src1 = NULL;
  return result;
}

struct tensor *dup_tensor(
  struct context * ctx,
  struct tensor *t){
  return new_tensor_impl(ctx, t->dtype, t->dimensions, t->number_elements, NULL);
}

void set_param(
  struct context *ctx,
  struct tensor *t){
  t->is_param = true;

  assert(t->grad == NULL);
  t->grad = dup_tensor(ctx, t);
}
float get_f32_1d(
  const struct tensor *t,
  size_t index){
  switch(t->dtype){
    case datatypes::FLOAT32:
      {
        assert(t->number_bytes[0] == sizeof(float));
        return ((float *)t->data)[index];
      }
      break;
    case datatypes::DOUBLE:
      {
        assert(t->number_bytes[0] == sizeof(double));
        return ((double *)t->data)[index];
      }
      break;
    case datatypes::INT:
      {
        assert(t->number_bytes[0] == sizeof(int));
        return ((int *)t->data)[index];
      }
      break;
    case datatypes::TYPE_COUNT:
      {
        assert(false);
      }
      break;
  }
  assert(false);
  return 0.0f;
}

struct tensor * set_f32(
  struct tensor *t,
  float value){
  const int n = nrows(t);
  const int nc = t->number_elements[0];
  const size_t n1 = t->number_bytes[1];

  char *const data = (char *)t->data;

  switch(t->dtype){
    case datatypes::FLOAT32:
      {
        // assert(t->number_bytes[0] == sizeof(float));
        for(int i = 0; i < n; i++){
          vec_set_f32(nc, (float*)(data+i*n1), value);
        }
      }
      break;
    case datatypes::INT:
      {
        assert(t->number_bytes[0] == sizeof(int));
        for(int i = 0; i < n; i++){
          vec_set_i32(nc, (int*)(data + i * n1), value);
        }
      }
      break;
    case datatypes::DOUBLE:
      {
        assert(t->number_bytes[0] == sizeof(double));
        for(int i = 0; i < n; i++){
          vec_set_d32(nc, (double*)(data + i * n1), value);
        }
      }
      break;
    case datatypes::TYPE_COUNT:
      {
        assert(false);
        // shouldn't happen
      }
      break;
  }
  return t;
}

// Math

// Addition
struct tensor * add_impl(
  struct context * ctx,
  struct tensor * a,
  struct tensor * b,
  bool inplace){
  //same_shape(a, b);
  bool is_node = false;

  if(!inplace && (a->grad||b->grad)){
    is_node = true;
  }

  struct tensor * result = inplace ? view_tensor(ctx, a) : dup_tensor(ctx, a);

  result->op  = ops::OP_ADD;
  result->grad = is_node ? dup_tensor(ctx, result) : NULL;
  result->src0 = a;
  result->src1 = b;

  return result;
}

struct tensor * add(
  struct context * ctx,
  struct tensor *a,
  struct tensor *b){
  return add_impl(ctx, a, b, false);
}

struct tensor * add_inplace(
  struct context * ctx,
  struct tensor *a,
  struct tensor *b){
  return add_impl(ctx, a, b, true);
}


