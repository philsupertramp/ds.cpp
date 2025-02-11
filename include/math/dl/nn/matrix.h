#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>  // for std::min
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
class matrix {
private:
    size_t rows, cols;
    std::vector<T> data; // Stored in row-major order

public:
    // Constructor: creates a matrix with dimensions (rows x cols) filled with 'initial'
    matrix(size_t rows, size_t cols, const T& initial = T())
      : rows(rows), cols(cols), data(rows * cols, initial) {}

    // Element access (mutable)
    inline T& operator()(size_t i, size_t j) {
        if (i >= rows || j >= cols)
            throw std::out_of_range("matrix index out of range");
        return data[i * cols + j];
    }

    // Element access (const)
    inline const T& operator()(size_t i, size_t j) const {
        if (i >= rows || j >= cols)
            throw std::out_of_range("matrix index out of range");
        return data[i * cols + j];
    }

    size_t numRows() const { return rows; }
    size_t numCols() const { return cols; }

    // matrix addition (elementwise)
    matrix<T> operator+(const matrix<T>& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Dimension mismatch in addition");
        matrix<T> result(rows, cols);
        const size_t total = rows * cols;
        for (size_t i = 0; i < total; ++i)
            result.data[i] = data[i] + other.data[i];
        return result;
    }

    // matrix subtraction (elementwise)
    matrix<T> operator-(const matrix<T>& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Dimension mismatch in subtraction");
        matrix<T> result(rows, cols);
        const size_t total = rows * cols;
        for (size_t i = 0; i < total; ++i)
            result.data[i] = data[i] - other.data[i];
        return result;
    }

    // Blocked matrix multiplication with optional OpenMP parallelization
    matrix<T> operator*(const matrix<T>& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("Dimension mismatch in multiplication");
        matrix<T> result(rows, other.cols, T()); // Zero-initialized

        const size_t m = rows, n = cols, p = other.cols;
        // Tunable block size; this value can be adjusted based on experiments.
        const size_t blockSize = 64;  

        // Parallelize the outer two loops if OpenMP is available.
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (size_t i0 = 0; i0 < m; i0 += blockSize) {
            for (size_t k0 = 0; k0 < n; k0 += blockSize) {
                for (size_t j0 = 0; j0 < p; j0 += blockSize) {
                    // Compute block boundaries.
                    size_t i_max = std::min(i0 + blockSize, m);
                    size_t k_max = std::min(k0 + blockSize, n);
                    size_t j_max = std::min(j0 + blockSize, p);
                    for (size_t i = i0; i < i_max; ++i) {
                        size_t resRowBase = i * p; // Base index for row i in the result.
                        for (size_t k = k0; k < k_max; ++k) {
                            T temp = data[i * n + k];  // Cache the value to reduce repeated access.
                            size_t otherRowBase = k * p; // Base index for row k in 'other'.
                            for (size_t j = j0; j < j_max; ++j) {
                                result.data[resRowBase + j] += temp * other.data[otherRowBase + j];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    // Utility method to print the matrix to the console.
    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j)
                std::cout << (*this)(i, j) << " ";
            std::cout << "\n";
        }
    }
};

