#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "utils.h"

__global__ void conv2d(uint8_t *N, float *F, uint8_t *P, int r, int width,
                       int height) {
  int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  float Pvalue = 0.0f;
  int row, col;

  for (int i = 0; i < 2 * r + 1; i++) {
    for (int j = 0; j < 2 * r + 1; j++) {
      row = outRow - r + i;
      col = outCol - r + j;

      if (row >= 0 && row < height && col >= 0 && col < width) {
        Pvalue += F[i * (2 * r + 1) + j] * N[row * width + col];
      }
    }
  }

  P[outRow * width + outCol] = (uint8_t)Pvalue;
}

void convolution(uint8_t *image, int width, int height) {
  uint8_t *image_d, *dst_d;
  int n = width * height;
  int size = n * sizeof(uint8_t);

  // Sobel Vertical
  // float filter[] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
  // Sobel Horizontal
  float filter[] = {1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f};
  float *filter_d;

  cudaMalloc((void **)&image_d, size);
  cudaMalloc((void **)&dst_d, size);
  cudaMalloc((void **)&filter_d, 9 * sizeof(float));

  cudaMemcpy(image_d, image, size, cudaMemcpyHostToDevice);
  cudaMemcpy(filter_d, filter, 9 * sizeof(float), cudaMemcpyHostToDevice);

  const int blockSize = 32;

  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(divCeil(width, blockSize), divCeil(height, blockSize), 1);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  conv2d<<<dimGrid, dimBlock>>>(image_d, filter_d, dst_d, 1, width, height);

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Time taken 3x3 filter: %.3f ms\n", milliseconds);

  cudaMemcpy(image, dst_d, size, cudaMemcpyDeviceToHost);

  cudaFree(image_d);
  cudaFree(dst_d);
}

int main() {
  int width, height, bpp;

  uint8_t *gray_image =
      stbi_load("public/output_gray.jpg", &width, &height, &bpp, 1);

  convolution(gray_image, width, height);
  stbi_write_jpg("public/output_filter.jpg", width, height, 1, gray_image, 90);

  stbi_image_free(gray_image);
  return 0;
}