#include "../Test.h"
#include <math/dl/tensor.h>


class TensorTestCase : public Test {
public:
  void run() {
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
  }
};

int main() {
  TensorTestCase().run();
  return 0;
}

