# Exercises chapter 6

2. For tiled matrix multiplication, of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will the kernel completely avoid uncoalesced accesses to global memory? (You need to consider only square blocks.)

    If the BLOCK_SIZE is a multiple of the warp size, every thread in a warp will load consecutive elements in the memory. If not, different threads in a warp may belong to differnte tiles and they can load non-consecutive data. So, to avoid uncoalesced access to global memory, the BLOCK_SIZE has to be a multiple of 32.

3. Consider the following CUDA kernel:
For each of the following memory accesses, specify whether they are coalesced or uncoalesced or coalescing is not applicable: 

    ```c
    01. __global__ void foo_kernel(float *a, float *b, float *c, float *d, float *e) {
    02.     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    03.     __shared__ float a_s[256];
    04.     __shared__ float bc_s[4 * 256];
    05.     a_s[threadIdx.x] = a[i];
    06.     for(unsigned int j = 0; j < 4; ++j) {
    07.         bc_s[j * 256 + threadIdx.x] = b[j * blockDim.x * gridDim.x + i] + c[i * 4 + j];
    08.     }
    09.     __syncthreads();
    10.     d[i + 8] = a_s[threadIdx.x];
    11.     e[i * 8] = bc_s[threadIdx.x * 4];
    12.}
    ```

    1. The access to array a of line 05

        Coalesced.

    2. The access to array a_s of line 05

        Not applicable.

    3. The access to array b of line 07

        Coalesced.

    4. The access to array c of line 07

        Uncoalesced

    5. The access to array bc_s of line 07

        Not applicable.

    6. The access to array a_s of line 10

        Not applicable.

    7. The access to array d of line 10

        Coalesced

    8. The access to array bc_s of line 11

        Not applicable.

    9. The access to array e of line 11

        Uncoalesced.

4. What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix multiplication kernels? 
    
    1. The simple kernel described in Chapter 3, Multidimensional Grids and Data, without any optimizations applied.
        ```c
        __global__ void matmul(float *m1, int r1, int c1, float *m2, int c2, float *dst) {
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
        ```

        There are 2 float operations (1 addition and 1 multiplication) and two float points reads. Assuming that we do not have to take writes in cosideration:

        $\frac{2 \text{ OP}}{2 \times 4 \text{ bytes}} = 0.25 \text{ OP/B}$

    2. The kernel described in Chapter 5, Memory Architecture and Data Locality, with shared memory tiling applied using a tile size of 32x32.
        ```c
        #define TILE_SIZE 32

        __global__ void tiled_matmul(float *m1, int r1, int c1, float *m2, int c2, float *dst) {
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
        ```

        For each iteration, there are $2$ reads of float point data and $2 \times \text{TILE\_SIZE}$ float point operations. Assuming that we do not have to take writes in cosideration:

        $\frac{2 \times 32 \text{ OP}}{2 \times 4\text{ bytes}} = \frac{64 \text{ OP}}{8\text{ bytes}} = 8 \text{ OP/B}$

    3. The kernel described in this chapter with shared memory tiling applied using a tile size of 32x32 and thread coarsening applied using a coarsening factor of 4.

        ```c
        #define TILE_SIZE 32
        #define COARSE_FACTOR 4

        __global__ void coarsed_tiled_matmul(float *m1, int r1, int c1, float *m2,
                                             int c2, float *dst) {
          __shared__ float tileM1[TILE_SIZE][TILE_SIZE];
          __shared__ float tileM2[TILE_SIZE][TILE_SIZE];

          int row = blockIdx.y * TILE_SIZE + threadIdx.y;
          int col = blockIdx.x * TILE_SIZE * COARSE_FACTOR + threadIdx.x;
          int number_tiles = (c1 + TILE_SIZE - 1) / TILE_SIZE;

          float sum[COARSE_FACTOR];
          for (int i = 0; i < COARSE_FACTOR; ++i) {
            sum[i] = 0.0f;
          }

          for (int i = 0; i < number_tiles; ++i) {
            int tile_row = i * TILE_SIZE + threadIdx.y;
            int tile_col = i * TILE_SIZE + threadIdx.x;

            tileM1[threadIdx.y][threadIdx.x] =
                (row < r1 && tile_col < c1) ? m1[row * c1 + tile_col] : 0.0f;

            for (int j = 0; j < COARSE_FACTOR; ++j) {
            
              tileM2[threadIdx.y][threadIdx.x] =
                  (col < c2 && tile_row < c1) ? m2[tile_row * c2 + col + j * TILE_SIZE]
                                              : 0.0f;

              __syncthreads();

              for (int k = 0; k < TILE_SIZE; ++k) {
                sum[j] += tileM1[threadIdx.y][k] * tileM2[k][threadIdx.x];
              }
              __syncthreads();
            }
          }

          for (int j = 0; j < COARSE_FACTOR; ++j) {
            if (row < r1 && col < c2) {
              dst[row * c2 + col + j * TILE_SIZE] = sum[j];
            }
          }
        }
        ```

        For each $1 + \text{COARSE\_FACTOR}$ loads, there are $2 \times \text{TILE\_SIZE} \times \text{COARSE\_FACTOR}$ float point operations. Assuming that we do not have to take writes in cosideration:
        
        $\frac{2 \times 32 \times 4 \text{ OP}}{5 \times 4 \text{ bytes}} = \frac{256 \text{ OP}}{20 \text{ bytes}} = 12.8 \text{ OP/B}$ 