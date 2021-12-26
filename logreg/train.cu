#include <stdio.h>

#include <cassert>
#include <utility>

#include "matrix.h"

const size_t BLOCK_DIM_Y = 512;
const size_t MAX_N_COLUMNS = 12288;
const size_t MAX_BATCH_SIZE = 12288;

__device__ float sigmoid(float z) {
    float s = 1.0 / (1.0 + expf(-z));
    return s;
}

void __global__ forward_kernel(Matrix& w, Matrix& b, Matrix& x, Matrix& y_hat) {
    __shared__ float sh_w[MAX_N_COLUMNS];

    size_t tid = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid < w.n_rows) {
        sh_w[tid] = w(tid, 0);
    }
    __syncthreads();

    if (tid < x.n_rows) {
        float acc = 0;
        for (size_t i_col = 0; i_col < x.n_cols; ++i_col) {
            acc += x(tid, i_col) * sh_w[i_col];
        }
        y_hat(tid, 0) = sigmoid(acc + b(0, 0));
    }
    __syncthreads();
}

void __global__ step_kernel(Matrix& x, Matrix& y, Matrix& y_hat, Matrix& w, Matrix& b, float lr) {
    __shared__ float y_diff[MAX_BATCH_SIZE];

    size_t tid = blockIdx.y * blockDim.y + threadIdx.y;
    if (tid < y.n_rows) {
        y_diff[tid] = y_hat(tid, 0) - y(tid, 0);
    }
    __syncthreads();

    if (tid < x.n_cols) {
        float wd = 0;
        float bd = 0;
        for (size_t i_row = 0; i_row < x.n_rows; ++i_row) {
            wd += x(i_row, tid) * y_diff[i_row];
            bd += y_diff[i_row];
        }
        w(tid, 0) -= lr * (wd / x.n_rows);
        b(0, 0) -= lr * (bd / x.n_rows);
    }
    __syncthreads();
}

int main() {
    size_t n_rows = 10000;
    size_t n_cols = 3000;
    const size_t n_epochs = 20;
    const size_t batch_size = 1024;
    const float lr = 0.01;
    float* x = create_random_data(n_rows * n_cols, 100);
    float* y = create_random_data(n_rows, 2);

    Matrix* x_batch_mat = new Matrix(batch_size, n_cols);
    Matrix* y_batch_mat = new Matrix(batch_size, 1);
    Matrix* w = new Matrix(x_batch_mat->n_cols, 1);
    Matrix* b = new Matrix(1, 1);
    Matrix* y_hat = new Matrix(batch_size, 1);
    Matrix* dw = new Matrix(x_batch_mat->n_cols, 1);
    assert(w->n_rows <= MAX_N_COLUMNS && x_batch_mat->n_rows <= MAX_BATCH_SIZE);

    for (size_t i_epoch = 0; i_epoch < n_epochs; ++i_epoch) {
        for (size_t row_from = 0; row_from < n_rows; row_from += batch_size) {
            size_t row_to = row_from + batch_size > n_rows ? n_rows : row_from + batch_size;
            size_t this_batch_size = row_to - row_from;
            float* x_batch = &x[row_from * n_cols];
            float* y_batch = &y[row_from];

            x_batch_mat->set_data(x_batch, this_batch_size);
            y_batch_mat->set_data(y_batch, this_batch_size);

            size_t grid_dim_y = (x_batch_mat->n_rows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
            forward_kernel<<<dim3(1, grid_dim_y), dim3(1, BLOCK_DIM_Y)>>>(*w, *b, *x_batch_mat,
                                                                          *y_hat);
            cudaDeviceSynchronize();

            step_kernel<<<dim3(1, grid_dim_y), dim3(1, BLOCK_DIM_Y)>>>(*x_batch_mat, *y_batch_mat,
                                                                       *y_hat, *w, *b, lr);
            cudaDeviceSynchronize();

            printf("i_epoch: %zu, i_step: %zu, y_hat[0]: %f, w[0]: %f\n", i_epoch,
                   row_from / batch_size, (*y_hat)(0, 0), (*w)(0, 0));
        }
    }

    delete x_batch_mat;
    free(x);
    free(y);

    return 0;
}
