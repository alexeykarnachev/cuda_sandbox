#include "matrix.h"

Matrix::Matrix(size_t n_rows, size_t n_cols)
    : n_rows(n_rows),
      n_cols(n_cols),
      n_values(n_rows * n_cols),
      size(sizeof(int) * n_rows * n_cols) {
    cudaMallocManaged(&data, size);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < n_values; ++i) {
        data[i] = rand() % 10;
    }
}

Matrix::~Matrix() { cudaFree(data); }

__host__ __device__ int& Matrix::operator()(size_t i_row, size_t i_col) {
    size_t idx = n_cols * i_row + i_col;
    return data[idx];
}
