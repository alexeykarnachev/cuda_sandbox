#include "utils.h"

void* Managed::operator new(size_t size) {
    void* ptr;
    cudaMallocManaged(&ptr, size);
    cudaDeviceSynchronize();
    return ptr;
}

void Managed::operator delete(void* ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
}

std::pair<dim3, dim3> get_grid_and_block_dims(std::pair<size_t, size_t> mat_dim,
                                              size_t block_side_size) {
    dim3 block_dim(block_side_size, block_side_size);

    size_t grid_dim_x = (mat_dim.second + block_dim.x - 1) / block_dim.x;
    size_t grid_dim_y = (mat_dim.first + block_dim.y - 1) / block_dim.y;
    dim3 grid_dim(grid_dim_x, grid_dim_y);

    return std::pair<dim3, dim3>(grid_dim, block_dim);
}
