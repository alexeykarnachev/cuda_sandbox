#include <stdio.h>

#include <cassert>

#include "matrix.h"

__global__ void mat_mul(Matrix* a, Matrix* b, Matrix* c) {
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_row < c->n_rows && c_col < c->n_cols) {
        int acc = 0;
        for (size_t i = 0; i < (*a).n_cols; ++i) {
            acc += (*a)(c_row, i) * (*b)(i, c_col);
        }
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
    std::pair<size_t, size_t> dim_a(100, 50);
    std::pair<size_t, size_t> dim_b(50, 200);
    std::pair<size_t, size_t> dim_c(dim_a.first, dim_b.second);

    Matrix* a = new Matrix(dim_a);
    Matrix* b = new Matrix(dim_b);
    Matrix* c = new Matrix(dim_c);
    assert(a->n_cols == b->n_rows && c->n_rows == a->n_rows && c->n_cols == b->n_cols);

    std::pair<dim3, dim3> dim = get_grid_and_block_dims(c->dim);
    mat_mul<<<dim.first, dim.second>>>(a, b, c);
    cudaDeviceSynchronize();

    validate_mul_on_cpu(*a, *b, *c);
    delete a, b, c;

    printf("SUCCESS!\n");
    return 0;
}
