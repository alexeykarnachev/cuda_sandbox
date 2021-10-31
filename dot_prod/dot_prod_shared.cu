#include <cassert>
#include <iostream>

#define imin(a, b) (a < b ? a : b);

const size_t N_VALUES = 1 << 25;
const size_t N_THREADS = 1 << 7;
const size_t N_BLOCKS = imin((1 << 16) - 1, (N_VALUES + N_THREADS - 1) / N_THREADS);

__global__ void dot_prod(int* a, int* b, int* d_block_results, size_t n_values) {
    __shared__ int redux_cache[N_THREADS];
    size_t local_thread_idx = threadIdx.x;
    size_t global_thread_idx = blockIdx.x * blockDim.x + local_thread_idx;
    if (global_thread_idx > n_values) {
        return;
    }
    int tmp = 0;
    while (global_thread_idx < n_values) {
        tmp += a[global_thread_idx] * b[global_thread_idx];
        global_thread_idx += blockDim.x * gridDim.x;
    }
    redux_cache[local_thread_idx] = tmp;
    __syncthreads();

    size_t redux_n_threads = N_THREADS;
    while (redux_n_threads > 1) {
        if (local_thread_idx < redux_n_threads / 2) {
            size_t i = local_thread_idx;
            size_t j = i + redux_n_threads / 2;
            redux_cache[i] += redux_cache[j];
        }
        redux_n_threads /= 2;
        __syncthreads();
    }

    if (local_thread_idx == 0) {
        d_block_results[blockIdx.x] = redux_cache[0];
    }
}

void init_values(int* x, size_t n_values) {
    for (size_t i = 0; i < n_values; ++i) {
        x[i] = rand() % 100;
    }
}

int sum_values(int* x, size_t n_values) {
    int result = 0;
    for (size_t i = 0; i < n_values; ++i) {
        result += x[i];
    }
    return result;
}

int dot_product_cpu(int* a, int* b, size_t n_values) {
    int result = 0;
    for (size_t i = 0; i < n_values; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void print_values(int* x, size_t n_values) {
    for (size_t i = 0; i < n_values; ++i) {
        std::cout << x[i] << " ";
    }
    std::cout << "\n";
}

int main() {
    size_t array_n_bytes = sizeof(int) * N_VALUES;
    size_t block_results_n_bytes = sizeof(int) * N_BLOCKS;

    int* a = (int*)malloc(array_n_bytes);
    int* b = (int*)malloc(array_n_bytes);
    init_values(a, N_VALUES);
    init_values(b, N_VALUES);

    int *d_a, *d_b, *d_block_results;
    cudaMalloc(&d_a, array_n_bytes);
    cudaMalloc(&d_b, array_n_bytes);
    cudaMalloc(&d_block_results, block_results_n_bytes);

    cudaMemcpy(d_a, a, array_n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, array_n_bytes, cudaMemcpyHostToDevice);

    dot_prod<<<N_BLOCKS, N_THREADS>>>(d_a, d_b, d_block_results, N_VALUES);
    int* block_results = (int*)malloc(block_results_n_bytes);
    cudaMemcpy(block_results, d_block_results, block_results_n_bytes, cudaMemcpyDeviceToHost);

    int cpu_result = dot_product_cpu(a, b, N_VALUES);
    int gpu_result = sum_values(block_results, N_BLOCKS);
    std::cout << "GPU result: " << gpu_result << std::endl;
    std::cout << "CPU result: " << cpu_result << std::endl;
    assert(gpu_result == cpu_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_block_results);
    delete[] a;
    delete[] b;
    delete[] block_results;

    std::cout << "SUCCESS!\n";
    return 0;
}
