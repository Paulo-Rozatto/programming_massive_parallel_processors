#include <cmath>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void add(float *a, float *b, float *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vecAdd(float *ah, float *bh, float *ch, int n) {
  int size = n * sizeof(float);
  float *ad, *bd, *cd;

  cudaMalloc((void **)&ad, size);
  cudaMalloc((void **)&bd, size);
  cudaMalloc((void **)&cd, size);

  cudaMemcpy(ad, ah, size, cudaMemcpyHostToDevice);
  cudaMemcpy(bd, bh, size, cudaMemcpyHostToDevice);

  add<<<ceil(n / 4.0), 4>>>(ad, bd, cd, size);

  cudaMemcpy(ch, cd, size, cudaMemcpyDeviceToHost);
  cudaFree(ah);
  cudaFree(ch);
  cudaFree(bh);
}

int main() {
  printf("Adding vecs: {1, 2, 3} + {4, 3, 2}\n");
  float a[] = {1, 2, 3};
  float b[] = {4, 3, 2};
  float c[] = {0, 0, 0};

  vecAdd(a, b, c, 3);
  printf("Result: { ");
  for (int i = 0; i < 3; i++) {
    printf("%.1f", c[i]);
    if (i < 2)
      printf(", ");
  }
  printf(" }\n");

  return 0;
}