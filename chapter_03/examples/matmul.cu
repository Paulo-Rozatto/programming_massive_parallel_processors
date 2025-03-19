#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

__global__ void matmul(float *m1, int r1, int c1, float *m2, int c2,
                       float *dst) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < r1 && col < c2) {
    float value = 0;
    for (int i = 0; i < c1; i++) {
      value += m1[row * c1 + i] * m2[i * c2 + col];
    }
    dst[row * c2 + col] = value;
  }
}

void mul(float *m1, int rows1, int cols1, float *m2, int rows2, int cols2,
         float *dst) {
  unsigned int size1 = rows1 * cols1 * sizeof(float);
  unsigned int size2 = rows2 * cols2 * sizeof(float);
  unsigned int size_dst = rows1 * cols2 * sizeof(float);

  float *m1_d, *m2_d, *dst_d;

  CHECK_CUDA(cudaMalloc((void **)&m1_d, size1));
  CHECK_CUDA(cudaMalloc((void **)&m2_d, size2));
  CHECK_CUDA(cudaMalloc((void **)&dst_d, size_dst));

  cudaMemcpy(m1_d, m1, size1, cudaMemcpyHostToDevice);
  cudaMemcpy(m2_d, m2, size2, cudaMemcpyHostToDevice);

  const int block_size = 32;
  dim3 blocks(block_size, block_size);
  dim3 grid(divCeil(cols2, block_size), divCeil(rows1, block_size));

  printf("%d\n", block_size);
  printf("%d - %d \n\n", divCeil(cols2, block_size),
         divCeil(rows1, block_size));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  matmul<<<grid, blocks>>>(m1_d, rows1, cols1, m2_d, cols2, dst_d);

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Time taken by matmul gpu: %.3f ms\n", milliseconds);

  cudaMemcpy(dst, dst_d, size_dst, cudaMemcpyDeviceToHost);

  cudaFree(m1_d);
  cudaFree(m2_d);
  cudaFree(dst_d);
}

void mul_cpu(float *m1, int r1, int c1, float *m2, int c2, float *dst) {
  clock_t start = clock();
  for (int row = 0; row < r1; row++) {
    for (int col = 0; col < c2; col++) {
      float value = 0;
      for (int i = 0; i < c1; i++) {
        value += m1[row * c1 + i] * m2[i * c2 + col];
      }
      dst[row * c2 + col] = value;
    }
  }
  clock_t end = clock();
  double milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;

  printf("Time taken by matmul cpu: %.3f ms\n", milliseconds);
}

float *create_random_matrix(int rows, int cols) {
  if (rows <= 0 || cols <= 0)
    return NULL;

  float *data = (float *)malloc(rows * cols * sizeof(float));

  for (int i = 0; i < rows * cols; i++) {
    data[i] = (float)rand() / (RAND_MAX + 1.0f) + 1.0f;
  }

  return data;
}

void printm(float *m, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    printf("[ ");
    for (int j = 0; j < cols; j++) {
      printf("%.2f ", m[i * cols + j]);
    }
    printf("]\n");
  }
}

int main() {
  int rows1 = 513, cols1 = 722;
  int rows2 = 722, cols2 = 130;
  rows1 = rows2 = cols1 = cols2 = 1024;

  float *m1 = create_random_matrix(rows1, cols1);
  float *m2 = create_random_matrix(rows2, cols2);
  float *dst = (float *)malloc(rows1 * cols2 * sizeof(float));

  for (int i = 0; i < rows1 * cols2; i++) {
    dst[i] = 0.0f;
  }

  mul(m1, rows1, cols1, m2, rows2, cols2, dst);
  mul_cpu(m1, rows1, cols1, m2, cols2, dst);

  // printm(m1, rows1, cols1);
  // printf("\n");
  // printm(m2, rows2, cols2);
  // printf("\n");
  // printm(dst, rows1, cols2);

  return 0;
}