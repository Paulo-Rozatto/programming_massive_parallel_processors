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

#define BLUR_SIZE 3

__global__ void boxblur(uint8_t *src, uint8_t *dst, int width, int height) {
  int col = (blockDim.x * blockIdx.x + threadIdx.x);
  int row = (blockDim.y * blockIdx.y + threadIdx.y);

  if (row < height && col < width) {
    int start_row = max(row - BLUR_SIZE, 0);
    int end_row = min(row + BLUR_SIZE + 1, height);

    int start_col = max(col - BLUR_SIZE, 0);
    int end_col = min(col + BLUR_SIZE + 1, width);

    int index;
    int r = 0, g = 0, b = 0;

    for (int i = start_row; i < end_row; i++) {
      for (int j = start_col; j < end_col; j++) {
        index = (i * width + j) * 3;
        r += src[index];
        g += src[index + 1];
        b += src[index + 2];
      }
    }

    int numberPixels = (end_col - start_col) * (end_row - start_row);
    index = (row * width + col) * 3;
    dst[index] = r / numberPixels;
    dst[index + 1] = g / numberPixels;
    dst[index + 2] = b / numberPixels;
  }
}

int divCeil(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

void blur(uint8_t *src, uint8_t *dst, int width, int height) {
  uint8_t *src_d, *dst_d;
  int size = width * height * 3 * sizeof(uint8_t);

  cudaMalloc((void **)&src_d, size);
  cudaMalloc((void **)&dst_d, size);

  cudaMemcpy(src_d, src, size, cudaMemcpyHostToDevice);

  const int blockSize = 32;

  dim3 dimBlock(blockSize, blockSize, 1);
  printf("%d - %d\n", divCeil(width, blockSize), divCeil(height, blockSize));
  dim3 dimGrid(divCeil(width, blockSize), divCeil(height, blockSize), 1);

  boxblur<<<dimGrid, dimBlock>>>(src_d, dst_d, width, height);

  cudaMemcpy(dst, dst_d, size, cudaMemcpyDeviceToHost);

  cudaFree(src_d);
  cudaFree(dst_d);
}

int main() {
  int width, height, bpp;

  uint8_t *input_image = stbi_load("public/memorial-nove-de-novembro.jpg",
                                   &width, &height, &bpp, 3);
  uint8_t *output_image =
      (uint8_t *)malloc(width * height * 3 * sizeof(uint8_t));

  for (int i = 0; i < width * height * 3; i++) {
    output_image[i] = 255;
  }

  printf("w: %d, h: %d, c: %d, s: %lu \n", width, height, bpp,
         sizeof(input_image));

  blur(input_image, output_image, width, height);

  stbi_write_jpg("public/output_blur.jpg", width, height, 3, output_image, 90);

  stbi_image_free(input_image);
  free(output_image);

  return 0;
}