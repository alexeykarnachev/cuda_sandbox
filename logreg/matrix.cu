#include "matrix.h"
#include <stdio.h>

Matrix::Matrix(size_t n_rows, size_t n_cols) : n_rows(n_rows), n_cols(n_cols) {
    size_t size = sizeof(float) * n_rows * n_cols;

    cudaMallocManaged(&data, size);
    cudaDeviceSynchronize();
};

void Matrix::set_data(float* source, size_t n_rows) {
    cudaMemcpy(data, source, sizeof(float) * n_rows * n_cols, cudaMemcpyHostToDevice);
}

Matrix::~Matrix() { cudaFree(data); };

__host__ __device__ float& Matrix::operator()(size_t i_row, size_t i_col) {
    size_t i_val = i_row * this->n_cols + i_col;
    return this->data[i_val];
}
