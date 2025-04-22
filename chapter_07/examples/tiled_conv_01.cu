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

#define FILTER_RADIUS 1
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)
#define FILTER_DIAMETER (2 * FILTER_RADIUS + 1)

__constant__ float F[FILTER_DIAMETER * FILTER_DIAMETER];

__global__ void conv2d(uint8_t *N, uint8_t *P, int width, int height) {
  int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
  int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

  __shared__ uint8_t N_s[IN_TILE_DIM][IN_TILE_DIM];

  if (row >= 0 && row < height && col >= 0 && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }

  __syncthreads();

  int tile_col = threadIdx.x - FILTER_RADIUS;
  int tile_row = threadIdx.y - FILTER_RADIUS;

  if (col >= 0 && col < width && row >= 0 && row < height) {
    if (tile_col >= 0 && tile_col < OUT_TILE_DIM && tile_row >= 0 &&
        tile_row < OUT_TILE_DIM) {
      float pvalue = 0.0f;

      for (int f_row = 0; f_row < FILTER_DIAMETER; f_row++) {
        for (int f_col = 0; f_col < FILTER_DIAMETER; f_col++) {
          pvalue += F[f_row * FILTER_DIAMETER + f_col] *
                    N_s[tile_row + f_row][tile_col + f_col];
        }
      }

      P[row * width + col] = pvalue;
    }
  }
}

void convolution(uint8_t *image, int width, int height) {
  uint8_t *image_d, *dst_d;
  int n = width * height;
  int size = n * sizeof(uint8_t);

  // Sobel Vertical
  // float filter[] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
  // Sobel Horizontal
  float filter[] = {1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f};

  cudaMalloc((void **)&image_d, size);
  cudaMalloc((void **)&dst_d, size);

  cudaMemcpy(image_d, image, size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F, filter,
                     FILTER_DIAMETER * FILTER_DIAMETER * sizeof(float));

  dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
  dim3 dimGrid(divCeil(width, IN_TILE_DIM), divCeil(height, IN_TILE_DIM), 1);

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  conv2d<<<dimGrid, dimBlock>>>(image_d, dst_d, width, height);

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
  printf("%d\n", FILTER_DIAMETER);

  uint8_t *gray_image =
      stbi_load("public/output_gray.jpg", &width, &height, &bpp, 1);

  convolution(gray_image, width, height);
  stbi_write_jpg("public/output_filter.jpg", width, height, 1, gray_image, 90);

  stbi_image_free(gray_image);
  return 0;
}