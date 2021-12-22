#include <stdio.h>

#include <cassert>

#include "matrix.h"

__global__ void mat_mul(Matrix* a, Matrix* b, Matrix* c) {
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;

    int acc = 0;
    for (size_t i = 0; i < (*a).n_cols; ++i) {
        acc += (*a)(c_row, i) * (*b)(i, c_col);
    }
    (*c)(c_row, c_col) = acc;
    printf("mat_mul on device done, c_row: %d, c_col: %d\n", c_row, c_col);
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
    printf("validate_mul_on_cpu on host done\n");
}

int main() {
    size_t a_rows = 64;
    size_t a_cols = 64;
    size_t b_rows = 64;
    size_t b_cols = 64;

    Matrix* a = new Matrix(a_rows, a_cols);
    Matrix* b = new Matrix(b_rows, b_cols);
    Matrix* c = new Matrix(a_rows, b_cols);

    mat_mul<<<dim3(4, 4), dim3(16, 16)>>>(a, b, c);
    cudaDeviceSynchronize();

    validate_mul_on_cpu(*a, *b, *c);
    delete a, b, c;
    printf("SUCCESS!\n");
    return 0;
}
