#pragma once
#include <memory>
#include <stdexcept>
#include <vector>
#include <numeric>


class Tensor {
  std::vector<size_t> _shape;
  std::vector<float> _data;

  size_t _size;

public:
  Tensor()
  {}

  explicit Tensor(const std::vector<size_t>& shape, float init_value = 0)
    : _shape(shape)
  {
    _size = std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>());
    _data = std::vector<float>(_size, init_value);
  }

  explicit Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
    : _shape(shape), _data(data)
  {
    _size = std::accumulate(begin(shape), end(shape), 1.0, std::multiplies<size_t>());
  }

  const std::vector<size_t> shape() const { return _shape; }

  size_t size() const { return _size; }

  void reshape(const std::vector<size_t>& new_shape){
    auto elems = std::accumulate(begin(new_shape), end(new_shape), 1.0, std::multiplies<size_t>());
    if(elems != _size){
      throw std::invalid_argument("New shape doesn't match element count of previous shape.");
    }
    _shape = new_shape;
  }

  // Non-const version: allows modification
  float& operator[](const std::vector<size_t>& indices) {
      size_t flat_index = compute_flat_index(indices);
      return _data[flat_index];
  }

  // Const version: for read-only access
  const float& operator[](const std::vector<size_t>& indices) const {
      size_t flat_index = compute_flat_index(indices);
      return _data[flat_index];
  }

  Tensor slice(const std::vector<size_t>& start, const std::vector<size_t>& end) const {
    // Ensure start and end dimensions are valid
    if (start.size() != _shape.size() || end.size() != _shape.size()) {
      throw std::invalid_argument("Start and end must have the same number of dimensions as the tensor.");
    }

    // Calculate the shape of the new tensor
    std::vector<size_t> new_shape(_shape.size());
    for (size_t i = 0; i < _shape.size(); ++i) {
      if (end[i] <= start[i] || end[i] > _shape[i]) {
        throw std::out_of_range("Invalid slice range.");
      }
      new_shape[i] = end[i] - start[i];
    }

    // Create a new tensor to store the slice
    Tensor sliced_tensor(new_shape);

    // Copy the data
    std::vector<size_t> current_index(new_shape.size(), 0);
    do {
      // Compute the index in the original tensor
      std::vector<size_t> original_index = start;
      for (size_t i = 0; i < current_index.size(); ++i) {
          original_index[i] += current_index[i];
      }

      // Copy the value
      sliced_tensor[current_index] = (*this)[original_index];

    } while (_increment_index(current_index, new_shape));

    return sliced_tensor;
  }
  Tensor operator+(const Tensor& other) const {
    if(other._size != _size){ throw std::invalid_argument("Wrong shapes."); }
    Tensor out(_shape, 0);
    for(size_t i = 0; i < _size; ++i){ out._data[i] = _data[i] + other._data[i]; }
    return out;
  }
  Tensor operator-(const Tensor& other) const {
    if(other._size != _size){ throw std::invalid_argument("Wrong shapes."); }
    Tensor out(_shape, 0);
    for(size_t i = 0; i < _size; ++i){ out._data[i] = _data[i] - other._data[i]; }
    return out;
  }
  Tensor operator*(const Tensor& other) const {
    // Check for broadcasting compatibility
    if (!is_broadcastable(this->_shape, other._shape)) {
      throw std::invalid_argument("Shapes are not broadcastable for multiplication");
    }

    // Compute the broadcasted shape
    std::vector<size_t> result_shape = broadcast_shape(this->shape(), other.shape());

    // Initialize result tensor
    Tensor result(result_shape);

    // Perform element-wise multiplication
    for (size_t i = 0; i < result.size(); ++i) {
      size_t this_idx = broadcast_index(i, this->shape(), result.shape());
      size_t other_idx = broadcast_index(i, other.shape(), result.shape());
      result._data[i] = this->_data[this_idx] * other._data[other_idx];
    }

    return result;
  }
  Tensor operator/(const Tensor& other) const {
    // Check for broadcasting compatibility
    if (!is_broadcastable(this->_shape, other._shape)) {
      throw std::invalid_argument("Shapes are not broadcastable for division");
    }

    // Compute the broadcasted shape
    std::vector<size_t> result_shape = broadcast_shape(this->shape(), other.shape());

    // Initialize result tensor
    Tensor result(result_shape);

    // Perform element-wise multiplication
    for (size_t i = 0; i < result.size(); ++i) {
      size_t this_idx = broadcast_index(i, this->shape(), result.shape());
      size_t other_idx = broadcast_index(i, other.shape(), result.shape());
      result._data[i] = this->_data[this_idx] / other._data[other_idx];
    }

    return result;
  }
  Tensor& operator+=(const Tensor& other) {
    if(other._size != _size){ throw std::invalid_argument("Wrong shapes."); }
    Tensor out(_shape, 0);
    for(size_t i = 0; i < _size; ++i){ _data[i] += other._data[i]; }
    return *this;
  }
  Tensor& operator-=(const Tensor& other){
    if(other._size != _size){ throw std::invalid_argument("Wrong shapes."); }
    Tensor out(_shape, 0);
    for(size_t i = 0; i < _size; ++i){ _data[i] -= other._data[i]; }
    return *this;
  }
  Tensor matmul(const Tensor& other) const {

    // Check shape compatibility for matrix multiplication
    if (_shape.size() != 2 || other._shape.size() != 2) {
      throw std::invalid_argument("matmul only supports 2D matrices.");
    }
    if (_shape[1] != other._shape[0]) {
      throw std::invalid_argument("Incompatible shapes for matrix multiplication.");
    }

    size_t m = _shape[0];  // Rows of A
    size_t n = _shape[1];  // Columns of A (Rows of B)
    size_t p = other._shape[1];  // Columns of B

    Tensor result({m, p}, 0.0f);

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < p; ++j) {
        // Compute the dot product of the i-th row of A and j-th column of B
        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
          sum += (*this)[{i, k}] * other[{k, j}];
        }
        result[{i, j}] = sum;
      }
    }
    return result;
  }

  // Method to transpose a tensor (swap axes)
  Tensor transpose(const std::vector<size_t>& axes) const {
    std::vector<size_t> new_shape(axes.size());
    for (size_t i = 0; i < axes.size(); ++i) {
      new_shape[i] = _shape[axes[i]];
    }
    
    Tensor transposed_tensor(new_shape);

    // Copying data to the transposed tensor
    size_t total_elements = _data.size();
    for (size_t i = 0; i < total_elements; ++i) {
      std::vector<size_t> original_indices = _get_indices(i);
      std::vector<size_t> transposed_indices(axes.size());
      for (size_t j = 0; j < axes.size(); ++j) {
        transposed_indices[j] = original_indices[axes[j]];
      }
      transposed_tensor[transposed_indices] = _data[i];
    }

    return transposed_tensor;
  }

  // Sum operation
  Tensor sum(int desired_axis, bool keep_dims = false) const {
    std::vector<size_t> new_shape = _shape;
    size_t axis = 0;
    if(desired_axis < 0){
      axis = static_cast<size_t>(_shape.size() + desired_axis);
    } else {
      axis = static_cast<size_t>(desired_axis);
    }
    size_t axis_size = _shape[axis];
    if (!keep_dims) {
      new_shape = {axis_size};
    }


    Tensor result(new_shape);  // New tensor for the result
    size_t stride = 1;
    // Compute the stride for each axis after the reduction
    for (size_t i = 0; i < axis; ++i) {
      stride *= _shape[i];
    }

    for (size_t i = 0; i < _size; ++i) {
      // Calculate the index along the axis we are summing over
      result._data[(i * stride) % axis_size] += _data[i];
    }

    return result;
  }
  Tensor mean(int desired_axis, bool keep_dims = false) const {
    std::vector<size_t> new_shape = _shape;
    size_t axis = 0;
    if(desired_axis < 0){
      axis = static_cast<size_t>(_shape.size() + desired_axis);
    } else {
      axis = static_cast<size_t>(desired_axis);
    }
    size_t axis_size = _shape[axis];
    if (!keep_dims) {
      new_shape = {axis_size};
    }


    Tensor result(new_shape);  // New tensor for the result
    size_t stride = 1;
    // Compute the stride for each axis after the reduction
    for (size_t i = 0; i < axis; ++i) {
      stride *= _shape[i];
    }

    for (size_t i = 0; i < _size; ++i) {
      // Calculate the index along the axis we are summing over
      result._data[(i * stride) % axis_size] += _data[i];
    }
    for(size_t ax = 0; ax < axis_size; ++ax){
      result._data[ax] /= stride;
    }

    return result;
  }
  Tensor max(size_t axis, bool keep_dims = false) const;
  Tensor argmax(size_t axis) const;

private:
  bool is_broadcastable(std::vector<size_t> shape1, std::vector<size_t> shape2) const {
    size_t dim1 = shape1.size();
    size_t dim2 = shape2.size();
    
    // Pad smaller shape with 1's at the beginning
    if (dim1 < dim2) {
        shape1.insert(shape1.begin(), dim2 - dim1, 1);
    } else if (dim2 < dim1) {
        shape2.insert(shape2.begin(), dim1 - dim2, 1);
    }
    
    // Check if shapes are compatible
    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
            return false;
        }
    }
    return true;
  }
  std::vector<size_t> broadcast_shape(std::vector<size_t> shape1, std::vector<size_t> shape2) const {
    size_t dim1 = shape1.size();
    size_t dim2 = shape2.size();

    // Pad smaller shape with 1's at the beginning
    if (dim1 < dim2) {
        shape1.insert(shape1.begin(), dim2 - dim1, 1);
    } else if (dim2 < dim1) {
        shape2.insert(shape2.begin(), dim1 - dim2, 1);
    }

    std::vector<size_t> result_shape;
    for (size_t i = 0; i < shape1.size(); ++i) {
        result_shape.push_back(std::max(shape1[i], shape2[i]));
    }
    return result_shape;
  }
  size_t broadcast_index(size_t index, const std::vector<size_t>& shape, const std::vector<size_t>& result_shape) const {
    size_t result_idx = 0;
    size_t stride = 1;
    
    for (int i = result_shape.size() - 1; i >= 0; --i) {
        size_t dim_size = result_shape[i];
        size_t shape_dim_size = shape[i];

        if (shape_dim_size == 1) {
            // If dimension size is 1, the index doesn't change
            result_idx += (index % dim_size) * stride;
        } else {
            // Otherwise, find the appropriate index in the original tensor
            result_idx += (index % dim_size) * stride;
        }

        stride *= dim_size;
        index /= dim_size;
    }

    return result_idx;
  }

  // Helper function to compute flat index (no const required here)
  size_t compute_flat_index(const std::vector<size_t>& indices) const {
      size_t flat_index = 0;
      size_t stride = 1;
      for (int i = indices.size() - 1; i >= 0; --i) {
          flat_index += indices[i] * stride;
          stride *= _shape[i];
      }
      return flat_index;
  }

  // Helper method to get the multi-dimensional indices from a flat index
  std::vector<size_t> _get_indices(size_t flat_index) const {
    std::vector<size_t> indices(_shape.size());
    size_t remainder = flat_index;
    for (int i = _shape.size() - 1; i >= 0; --i) {
      indices[i] = remainder % _shape[i];
      remainder /= _shape[i];
    }
    return indices;
  }
  // Helper function to increment a multi-dimensional index
  bool _increment_index(std::vector<size_t>& index, const std::vector<size_t>& shape) const {
    for (int i = index.size() - 1; i >= 0; --i) {
        if (++index[i] < shape[i]) {
            return true;
        }
        index[i] = 0;
    }
    return false;
  }

public:
  friend std::ostream& operator<<(std::ostream& ostr, const Tensor& t){
    ostr << "Tensor([";
    for (size_t i = 0; i < t._data.size(); ++i) {
      ostr << t._data[i] << " ";
      if (t._shape.size() > 1 && (i + 1) % t._shape[1] == 0) {
        ostr << std::endl;
      }
    }
    ostr << "], shape=[";
    for(auto const& e : t._shape){
      ostr << e << ",";
    }
    ostr << "])";
    return ostr;
  }
};

