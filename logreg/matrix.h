#include "utils.h"

class Matrix : public Managed {
   public:
    float* data;
    size_t n_rows;
    size_t n_cols;

    Matrix(size_t n_rows, size_t n_cols);
    ~Matrix();
    
    void set_data(float* source, size_t n_rows);
    __host__ __device__ float& operator()(size_t i_row, size_t i_col);
};
