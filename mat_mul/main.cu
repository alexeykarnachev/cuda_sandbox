#include <stdio.h>

#include <cassert>

#include "matrix.h"

const size_t BLOCK_SIDE_SIZE = 16;

__global__ void mat_mul(Matrix* a, Matrix* b, Matrix* c) {
    int c_row = blockIdx.y * BLOCK_SIDE_SIZE + threadIdx.y;
    int c_col = blockIdx.x * BLOCK_SIDE_SIZE + threadIdx.x;
    if (c_row < c->n_rows && c_col < c->n_cols) {
        int acc = 0;
        for (size_t i = 0; i < (*a).n_cols; ++i) {
            acc += (*a)(c_row, i) * (*b)(i, c_col);
        }
        (*c)(c_row, c_col) = acc;
    }
}

__global__ void mat_mul_shared(Matrix* a, Matrix* b, Matrix* c) {
    __shared__ int sa[BLOCK_SIDE_SIZE][BLOCK_SIDE_SIZE];
    __shared__ int sb[BLOCK_SIDE_SIZE][BLOCK_SIDE_SIZE];

    int c_row = blockIdx.y * BLOCK_SIDE_SIZE + threadIdx.y;
    int c_col = blockIdx.x * BLOCK_SIDE_SIZE + threadIdx.x;

    int acc = 0;
    for (size_t offset = 0; offset < a->n_cols; offset += BLOCK_SIDE_SIZE) {
        int a_row = c_row;
        int a_col = offset + threadIdx.x;
        int b_row = offset + threadIdx.y;
        int b_col = c_col;

        if (a_row < a->n_rows && a_col < a->n_cols) {
            sa[threadIdx.y][threadIdx.x] = (*a)(a_row, a_col);
        } else {
            sa[threadIdx.y][threadIdx.x] = 0;
        }
        if (b_row < b->n_rows && b_col < b->n_cols) {
            sb[threadIdx.y][threadIdx.x] = (*b)(b_row, b_col);
        } else {
            sb[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (size_t j = 0; j < BLOCK_SIDE_SIZE; ++j) {
            acc += sa[threadIdx.y][j] * sb[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (c_row < c->n_rows && c_col < c->n_cols) {
        (*c)(c_row, c_col) = acc;
    }
}

void validate_mul_on_cpu(Matrix& a, Matrix& b, Matrix& c) {
    for (int a_row = 0; a_row < a.n_rows; ++a_row) {
        for (int b_col = 0; b_col < b.n_cols; ++b_col) {
            int acc = 0;
            for (int i = 0; i < b.n_rows; ++i) {
                acc += a(a_row, i) * b(i, b_col);
            }
            assert(acc == c(a_row, b_col));
        }
    }
}

int main() {
    std::pair<size_t, size_t> dim_a(200, 100);
    std::pair<size_t, size_t> dim_b(100, 30);
    std::pair<size_t, size_t> dim_c(dim_a.first, dim_b.second);

    Matrix* a = new Matrix(dim_a);
    Matrix* b = new Matrix(dim_b);
    Matrix* c = new Matrix(dim_c);
    assert(a->n_cols == b->n_rows && c->n_rows == a->n_rows && c->n_cols == b->n_cols);
    std::pair<dim3, dim3> dim = get_grid_and_block_dims(c->dim, BLOCK_SIDE_SIZE);

    // mat_mul<<<dim.first, dim.second>>>(a, b, c);
    // cudaDeviceSynchronize();

    mat_mul_shared<<<dim.first, dim.second>>>(a, b, c);
    cudaDeviceSynchronize();

    validate_mul_on_cpu(*a, *b, *c);
    delete a, b, c;

    printf("SUCCESS!\n");
    return 0;
}
