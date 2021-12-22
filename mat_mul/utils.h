#include <utility>

class Managed {
   public:
    void* operator new(size_t size);
    void operator delete(void* ptr);
};

std::pair<dim3, dim3> get_grid_and_block_dims(std::pair<size_t, size_t> mat_dim);
