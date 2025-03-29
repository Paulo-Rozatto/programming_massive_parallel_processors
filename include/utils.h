#ifndef UTILS_PMPP_H
#define UTILS_PMPP_H

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),     \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

inline int divCeil(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

#endif