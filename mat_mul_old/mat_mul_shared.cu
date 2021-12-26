#include <cassert>
#include <iostream>

const size_t MAT_DIM = 1024;
const size_t MAT_N_BYTES = sizeof(int) * MAT_DIM * MAT_DIM;
const size_t BLOCK_DIM = 32;
const size_t GRID_DIM = (MAT_DIM + BLOCK_DIM - 1) / BLOCK_DIM;

__global__ void mat_mul(int* d_a, int* d_b, int* d_c) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int row = bid_y * BLOCK_DIM + tid_y;
    int col = bid_x * BLOCK_DIM + tid_x;

    __shared__ int sm_a[BLOCK_DIM][BLOCK_DIM];
    __shared__ int sm_b[BLOCK_DIM][BLOCK_DIM];

    int tmp = 0;
    for (size_t i = 0; i < MAT_DIM; i += BLOCK_DIM) {
        sm_a[tid_y][tid_x] = d_a[row * MAT_DIM + i + tid_x];
        sm_b[tid_y][tid_x] = d_b[col + MAT_DIM * i + MAT_DIM * tid_y];
        __syncthreads();

        for (size_t j = 0; j < BLOCK_DIM; ++j) {
            tmp += sm_a[tid_y][j] * sm_b[j][tid_x];
        }
        __syncthreads();
    }
    d_c[row * MAT_DIM + col] = tmp;
}

void init_values(int* m, size_t mat_dim) {
    for (size_t i = 0; i < mat_dim * mat_dim; ++i) {
        m[i] = rand() % 10;
    }
}

void validate_mat_mul(int* a, int* b, int* c, size_t mat_dim) {
    for (size_t i = 0; i < mat_dim; ++i) {
        for (size_t j = 0; j < mat_dim; ++j) {
            int tmp = 0;
            for (size_t k = 0; k < mat_dim; ++k) {
                tmp += a[mat_dim * i + k] * b[j + k * mat_dim];
            }
            assert(tmp == c[i * mat_dim + j]);
        }
    }
}

int main() {
    int* h_a = (int*)malloc(MAT_N_BYTES);
    int* h_b = (int*)malloc(MAT_N_BYTES);
    int* h_c = (int*)malloc(MAT_N_BYTES);
    init_values(h_a, MAT_DIM);
    init_values(h_b, MAT_DIM);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, MAT_N_BYTES);
    cudaMalloc(&d_b, MAT_N_BYTES);
    cudaMalloc(&d_c, MAT_N_BYTES);
    cudaMemcpy(d_a, h_a, MAT_N_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, MAT_N_BYTES, cudaMemcpyHostToDevice);

    mat_mul<<<dim3(GRID_DIM, GRID_DIM), dim3(BLOCK_DIM, BLOCK_DIM)>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, MAT_N_BYTES, cudaMemcpyDeviceToHost);
    validate_mat_mul(h_a, h_b, h_c, MAT_DIM);

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    std::cout << "SUCCESS!\n";

    return 0;
}
