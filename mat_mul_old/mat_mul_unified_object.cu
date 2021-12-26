#include <stdio.h>

#include <cassert>

class Managed {
   public:
    void* operator new(size_t size) {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

class Matrix : public Managed {
   public:
    size_t n_rows;
    size_t n_cols;

    Matrix(size_t n_rows, size_t n_cols)
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

    __host__ __device__ int& operator()(int i_row, int i_col) {
        size_t idx = n_cols * i_row + i_col;
        return data[idx];
    }

    ~Matrix() { cudaFree(data); }

   private:
    size_t n_values;
    size_t size;
    int* data;
};

__global__ void mat_mul(Matrix* a, Matrix* b, Matrix* c) {
    int c_row = blockIdx.y * blockDim.y + threadIdx.y;
    int c_col = blockIdx.x * blockDim.x + threadIdx.x;

    int acc = 0;
    for (size_t i = 0; i < (*a).n_cols; ++i) {
        acc += (*a)(c_row, i) * (*b)(i, c_col);
    }
    (*c)(c_row, c_col) = acc;
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
    return 0;
}
