nvcc -dc *.cu && \
nvcc *.o -o train.out && \
rm ./*.o && \
./train.out
