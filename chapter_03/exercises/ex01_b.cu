#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

/*
 * Write a kernel that has each thread produce one output matrix column.
 * Fill in the execution configuration parameters for the design.
 */

__global__ void mulcol(float *m1, int r1, int c1, float *m2, int c2,
                       float *dst) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < c2) {
    for (int row = 0; row < r1; row++) {
      float value = 0;

      for (int i = 0; i < c1; i++) {
        value += m1[row * c1 + i] * m2[i * c2 + col];
      }
      dst[row * c2 + col] = value;
    }
  }
}

void mulcol(float *m1, int rows1, int cols1, float *m2, int rows2, int cols2,
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

  const int block_size = 256;
  int grid = divCeil(cols2, block_size);

  printf("%d\n", block_size);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  mulcol<<<grid, block_size>>>(m1_d, rows1, cols1, m2_d, cols2, dst_d);

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

void matmul_cpu(float *m1, int rows1, int cols1, float *m2, int rows2,
                int cols2, float *dst) {
  clock_t start = clock();
  for (int i = 0; i < rows1; i++) {
    for (int j = 0; j < cols2; j++) {
      float value = 0;
      for (int k = 0; k < rows2; k++) {
        value += m1[i * cols1 + k] * m2[k * cols2 + j];
      }
      dst[i * cols2 + j] = value;
    }
  }
  clock_t end = clock();
  double milliseconds = 1000.0 * (end - start) / CLOCKS_PER_SEC;

  printf("Time taken by matmul cpu: %.3f ms\n", milliseconds);
}

float *create_random_matrix(int rows, int cols, float v) {
  if (rows <= 0 || cols <= 0)
    return NULL;

  float *data = (float *)malloc(rows * cols * sizeof(float));

  for (int i = 0; i < rows * cols; i++) {
    // data[i] = (float)rand() / (RAND_MAX + 1.0f) + 1.0f;
    data[i] = v;
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

  float *m1 = create_random_matrix(rows1, cols1, 1.0f);
  float *m2 = create_random_matrix(rows2, cols2, 2.0f);
  float *dst = (float *)malloc(rows1 * cols2 * sizeof(float));

  for (int i = 0; i < rows1 * cols2; i++) {
    dst[i] = 0.0f;
  }

  mulcol(m1, rows1, cols1, m2, rows2, cols2, dst);
  // matmul_cpu(m1, rows1, cols1, m2, rows2, cols2, dst);

  // printm(m1, rows1, cols1);
  // printf("\n");
  // printm(m2, rows2, cols2);
  // printf("\n");
  // printm(dst, rows1, cols2);

  return 0;
}