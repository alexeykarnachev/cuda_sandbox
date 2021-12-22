#include "utils.h"
#include <utility>

class Matrix : public Managed {
   public:
    size_t n_rows;
    size_t n_cols;
    std::pair<size_t, size_t> dim;

    Matrix(std::pair<size_t, size_t> dim);
    ~Matrix();

    __host__ __device__ int& operator()(size_t i_row, size_t i_col);

   private:
    size_t n_values;
    size_t size;
    int* data;
};
