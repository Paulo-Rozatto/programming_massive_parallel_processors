#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdlib.h>
#include <time.h>

#include "utils.h"

#define TILE_SIZE 16

__global__ void tiled_matmul(float *m1, int r1, int c1, float *m2, int c2,
                             float *dst) {
  __shared__ float tileM1[TILE_SIZE][TILE_SIZE];
  __shared__ float tileM2[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  int number_tiles = (c1 + TILE_SIZE - 1) / TILE_SIZE;

  float sum = 0.0f;

  for (int i = 0; i < number_tiles; ++i) {
    int tile_row = i * TILE_SIZE + threadIdx.y;
    int tile_col = i * TILE_SIZE + threadIdx.x;

    tileM1[threadIdx.y][threadIdx.x] =
        (row < r1 && tile_col < c1) ? m1[row * c1 + tile_col] : 0.0f;

    tileM2[threadIdx.y][threadIdx.x] =
        (col < c2 && tile_row < c1) ? m2[tile_row * c2 + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tileM1[threadIdx.y][k] * tileM2[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < r1 && col < c2) {
    dst[row * c2 + col] = sum;
  }
}

void mul(float *m1_h, int r1, int c1, float *m2_h, int c2, float *dst_h) {

  float *m1_d, *m2_d, *dst_d;
  size_t sizeM1 = r1 * c1 * sizeof(float);
  size_t sizeM2 = c1 * c2 * sizeof(float);
  size_t sizeDst = r1 * c2 * sizeof(float);

  cudaMalloc(&m1_d, sizeM1);
  cudaMalloc(&m2_d, sizeM2);
  cudaMalloc(&dst_d, sizeDst);

  cudaMemcpy(m1_d, m1_h, sizeM1, cudaMemcpyHostToDevice);
  cudaMemcpy(m2_d, m2_h, sizeM2, cudaMemcpyHostToDevice);

  dim3 dimBlock(TILE_SIZE, TILE_SIZE);
  dim3 dimGrid(divCeil(c1, TILE_SIZE), divCeil(r1, TILE_SIZE));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  tiled_matmul<<<dimGrid, dimBlock>>>(m1_d, r1, c1, m2_d, c2, dst_d);

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Time taken by matmul gpu: %.3f ms\n", milliseconds);

  cudaMemcpy(dst_h, dst_d, sizeDst, cudaMemcpyDeviceToHost);

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
  // rows1 = rows2 = cols1 = cols2 = 1024;

  float *m1 = create_random_matrix(rows1, cols1);
  float *m2 = create_random_matrix(rows2, cols2);
  float *dst = (float *)malloc(rows1 * cols2 * sizeof(float));

  for (int i = 0; i < rows1 * cols2; i++) {
    dst[i] = 0.0f;
  }

  mul(m1, rows1, cols1, m2, cols2, dst);
  // mul_cpu(m1, rows1, cols1, m2, cols2, dst);

  // printm(m1, rows1, cols1);
  // printf("\n");
  // printm(m2, rows2, cols2);
  // printf("\n");
  // printm(dst, rows1, cols2);

  return 0;
}