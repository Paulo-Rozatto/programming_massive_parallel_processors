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

__global__ void rgb2gray(uint8_t *rgb, uint8_t *gry, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = i * 3;

  if (i < n) {
    gry[i] = .21f * rgb[j] + .72f * rgb[j + 1] + .07f * rgb[j + 2];
  }
}

void cvtGrayscale(uint8_t *rgb, uint8_t *gry, int width, int height) {
  uint8_t *rgb_d, *gry_d;
  int n = width * height;
  int size = n * sizeof(uint8_t);

  cudaMalloc((void **)&rgb_d, size * 3);
  cudaMalloc((void **)&gry_d, size);

  cudaMemcpy(rgb_d, rgb, size * 3, cudaMemcpyHostToDevice);

  rgb2gray<<<(n + 1024 - 1) / 1024, 1024>>>(rgb_d, gry_d, n);

  cudaMemcpy(gry, gry_d, size, cudaMemcpyDeviceToHost);

  cudaFree(rgb_d);
  cudaFree(gry_d);
}

int main() {
  int width, height, bpp;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

  uint8_t *rgb_image = stbi_load("public/memorial-nove-de-novembro.jpg", &width,
                                 &height, &bpp, 3);

  uint8_t *gray_image = (uint8_t *)malloc(width * height * sizeof(uint8_t));

  printf("w: %d, h: %d, c: %d, s: %lu \n", width, height, bpp,
         sizeof(rgb_image));

  int n = width * height * bpp;
  printf("%d\n", n);
  cvtGrayscale(rgb_image, gray_image, width, height);
  stbi_write_jpg("public/output_gray.jpg", width, height, 1, gray_image, 90);

  stbi_image_free(rgb_image);
  free(gray_image);

  return 0;
}