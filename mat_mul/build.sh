nvcc -dc *.cu && \
nvcc *.o -o mat_mul_shared.out && \
rm ./*.o && \
./mat_mul_shared.out
