#include "../Test.h"
#include <math/dl/tensor.h>


class TensorTestCase : public Test {
public:
  /*
   * %TODO:
   * - nrows
   * - dup
   * - dup_inplace
   * - set_param
   * - set_f32
   * - get_f32_1d
   * - add
   * - add_inplace
   * - print_objects
   * - print_object
   * - vec_set_i32
   * - vec_set_f32
   * - vec_set_d32
   */

  bool TestConstructors() {
    struct init_params params = {
      .mem_size = 128 * 1024 * 1024,
      .mem_buffer = NULL,
    };

    struct context* ctx0 = tensor_init(params);

    struct tensor * t1 = new_tensor_1d(ctx0, datatypes::FLOAT32, 10);
    struct tensor * t2 = new_tensor_2d(ctx0, datatypes::FLOAT32, 10, 20);
    struct tensor * t3 = new_tensor_3d(ctx0, datatypes::FLOAT32, 10, 20, 30);
    struct tensor * t4 = new_tensor_4d(ctx0, datatypes::FLOAT32, 1, 2, 3, 4);

    AssertEqual(t1->dimensions, 1);
    AssertEqual(t1->number_elements[0], 10);
    AssertEqual(t1->number_bytes[1], 10 * sizeof(float));

    AssertEqual(t2->dimensions, 2);
    AssertEqual(t2->number_elements[0], 10);
    AssertEqual(t2->number_elements[1], 20);
    AssertEqual(t2->number_bytes[1], 10 * sizeof(float));
    AssertEqual(t2->number_bytes[2], 10 * 20 * sizeof(float));

    AssertEqual(t3->dimensions, 3);
    AssertEqual(t3->number_elements[0], 10);
    AssertEqual(t3->number_elements[1], 20);
    AssertEqual(t3->number_elements[2], 30);
    AssertEqual(t3->number_bytes[1], 10 * sizeof(float));
    AssertEqual(t3->number_bytes[2], 10 * 20 * sizeof(float));
    AssertEqual(t3->number_bytes[3], 10 * 20 * 30 * sizeof(float));

    AssertEqual(t4->dimensions, 4);
    AssertEqual(t4->number_elements[0], 1);
    AssertEqual(t4->number_elements[1], 2);
    AssertEqual(t4->number_elements[2], 3);
    AssertEqual(t4->number_elements[3], 4);
    AssertEqual(t4->number_bytes[1], 1 * sizeof(float));
    AssertEqual(t4->number_bytes[2], 1 * 2 * sizeof(float));
    AssertEqual(t4->number_bytes[3], 1 * 2 * 3 * sizeof(float));
    AssertEqual(t4->number_bytes[4], 1 * 2 * 3 * 4 * sizeof(float));

    tensor_free(ctx0);
    return true;
  }

  bool TestAddition(){
    struct init_params params = {
      .mem_size = 128 * 1024 * 1024,
      .mem_buffer = NULL,
    };

    struct context* ctx0 = tensor_init(params);
    struct tensor * x = new_tensor_1d(ctx0, datatypes::FLOAT32, 1);

    set_param(ctx0, x);

    struct tensor * a = new_tensor_1d(ctx0, datatypes::FLOAT32, 1);

    // a + x
    struct tensor * res = add(ctx0, a, x);

    print_objects(ctx0);

    // gf = g(x) = a + x
    struct cgraph gf = build_forward(res);
    // gb = g(x)' = 1.0
    struct cgraph gb = build_backward(ctx0, &gf, false);

    set_f32(x, 2.0f);
    set_f32(a, 5.0f);

    graph_reset(&gf);

    // (a + x)' == 1.0
    set_f32(res->grad, 1.0f);

    graph_compute(ctx0, &gb);

    printf("f     = %f\n", get_f32_1d(res, 0));
    printf("df/dx = %f\n", get_f32_1d(x->grad, 0));

    // assertions

    return true;
  }

  void run(){
    TestConstructors();
    TestAddition();
  }
};

int main() {
  TensorTestCase().run();
  return 0;
}

