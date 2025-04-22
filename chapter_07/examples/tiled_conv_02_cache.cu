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
#define TILE_DIM 32
#define FILTER_DIAMETER (2 * FILTER_RADIUS + 1)

__constant__ float F[FILTER_DIAMETER * FILTER_DIAMETER];

__global__ void conv2d(uint8_t *N, uint8_t *P, int width, int height) {
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;

  __shared__ uint8_t N_s[TILE_DIM][TILE_DIM];

  if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }

  __syncthreads();

  if (col < width && row < height) {
    float pvalue = 0.0f;

    for (int f_row = 0; f_row < FILTER_DIAMETER; f_row++) {
      for (int f_col = 0; f_col < FILTER_DIAMETER; f_col++) {

        // use shared memory
        if (threadIdx.x - FILTER_RADIUS + f_col >= 0 &&
            threadIdx.x - FILTER_RADIUS + f_col < TILE_DIM &&
            threadIdx.y - FILTER_RADIUS + f_row >= 0 &&
            threadIdx.y - FILTER_RADIUS + f_row < TILE_DIM) {
          pvalue += F[f_row * FILTER_DIAMETER + f_col] *
                    N_s[threadIdx.y - FILTER_RADIUS + f_row]
                       [threadIdx.x - FILTER_RADIUS + f_col];
        }
        // rely on L2 cache
        else if (row - FILTER_RADIUS + f_row >= 0 &&
                 row - FILTER_RADIUS + f_row < height &&
                 col - FILTER_RADIUS + f_col >= 0 &&
                 col - FILTER_RADIUS + f_col < width) {
          pvalue += F[f_row * FILTER_DIAMETER + f_col] *
                    N[(row - FILTER_RADIUS + f_row) * width + col -
                      FILTER_RADIUS + f_col];
        }
      }
    }

    P[row * width + col] = pvalue;
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

  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  dim3 dimGrid(divCeil(width, TILE_DIM), divCeil(height, TILE_DIM), 1);

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