# Exercises chapter 4

1. Consider the following CUDA kernel and the corresponding host function that calls it:
    ```c
    01. __global__ void foo_kernel(int *a, int *b) {
    02.     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    03.     if (threadIdx.x < 40 || threadIdx.x >= 104) {
    04.         b[i] = a[i] + 1;
    05.     }
    06.     if (i % 2 == 0) {
    07.         a[i] = b[i] * 2;
    08.     }
    09.     for (unsigned int j = 0; j < 5 - (i%3); ++j) {
    10.         b[i] += j;
    11.     }
    12. }
    13. void foo(int *a_d, int *b_d) {
    14.     unsigned int N = 1024;
    15.     foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d);
    16. }
    ```

    1. What is the number of warps per block?
        
        $128 \div 32 = 4$.

    2. What is the number of warps in the grid?

        $\frac{(1024 + 128 - 1)}{128} \times 4 = 32$

    3. For the statement on line 04:
        1. How many warps in the grid are active?
            
            In each grid, the first warp (0,31), the second warp (32, 63), and the fourth warp (96, 127) are active.

            So it's active $\frac{3}{4} \times 32 = 24$ warps.

        2. How many warps in the grid are divergent?

            Half of them, 16 warps.

        3. What is the SIMD efficiency (in %) of warp 0 of block 0? 

            $\frac{(32)}{32} * 100 = 100\%$

        4. What is the SIMD efficiency (in %) of warp 1 of block 0? 

            $\frac{(24)}{32} * 100 = 75\%$

        5. What is the SIMD efficiency (in %) of warp 3 of block 0?

             $\frac{(24)}{32} * 100 = 75\%$

    4. For the statement on line 07: 
        1. How many warps in the grid are active? 

            All of them are active.

        2. How many warps in the grid are divergent? 

            All of them are divergent.

        3. What is the SIMD efficiency (in %) of warp 0 of block 0?

            $\frac{64}{128} \text{ active threads in block 0} \times 100 = 50\%$

    5. For the loop on line 09: 
        1. How many iterations have no divergence? 
            
            The expression $(i \% 3)$ can assume values 0, 1 or 2. So iterations where $[i < (5 - 2)]$ always execute. So there are 2 iterations without divergence.

        2. How many iterations have divergence?

            Three iterations have divergence.

2. For a vector addition, assume that the vector length is 2000, each thread calculates one output element, and the thread block size is 512 threads. How many threads will be in the grid?

    $\lceil 2000 \div 512 \rceil = 4 \text{ blocks in grid}$;
    
    $4 \times 512 = 2048 \text{ threads in grid}$.

3. For the previous question, how many warps do you expect to have divergence due to the boundary check on vector length?

    Only the one warp which will calculate the positions 1984 to 2015.


4. Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and 2.9; they spend the rest of their time waiting for the barrier. What percentage of the threads’ total execution time is spent waiting for the barrier? 

    $(3.0 - 2.0) + (3.0 - 2.3) + (3.0 - 3.0) + (3.0 - 2.8) + (3.0 - 2.4) + (3.0 - 1.9) + (3.0 - 2.6) + (3.0 - 2.9)= 4.1 \mu s$

    $4.1 \mu s \div 24 \mu s \times 100 = 17.08\%$

5. A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave out the __syncthreads() instruction wherever barrier synchronization is needed. Do you think this is a good idea? Explain. 

   No, it is not. While a single warp executes in lockstep, divergent branches and memory access delays can cause inconsistencies. Also, there's no guarantee that the warp size will remain 32 threads in the future hardware releases.


6. If a CUDA device’s SM can take up to 1536 threads and up to 4 thread blocks, which of the following block configurations would result in the most number of threads in the SM? 
    1. 128 threads per block 
    2. 256 threads per block 
    3. 512 threads per block   
    4. 1024 threads per block  

    Option 3. The device can handle 3 blocks with 512 threads per blocks. That uses all threads avalable in the SM.

7. Assume a device that allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM are possible. In the cases in which it is possible, indicate the occupancy level. 
    1. 8 blocks with 128 threads each 

        $8 \times 128 = 1024 \text{ threads}$, possible.

        $\frac{1024}{2048} \times 100 = 50\% \text{ occupancy}$.
    2. 16 blocks with 64 threads each 

        $16 \times 64 = 1024 \text{ threads}$, possible.
        
        $\frac{1024}{2048} \times 100 = 50\% \text{ occupancy}$.

    3. 32 blocks with 32 threads each

        $32 \times 32 = 1024 \text{ threads}$, possible.
        
        $\frac{1024}{2048} \times 100 = 50\% \text{ occupancy}$.

    4. 64 blocks with 32 threads each 

        $64 \times 32 = 2028 \text{ threads}$, possible.
        
        $\frac{2048}{2048} \times 100 = 100\% \text{ occupancy}$.

    5. 32 blocks with 64 threads each

        $64 \times 32 = 2028 \text{ threads}$, possible.
        
        $\frac{2048}{2048} \times 100 = 100\% \text{ occupancy}$.

8. Consider a GPU with the following hardware limits: 2048 threads per SM, 32 blocks per SM, and 64K (65,536) registers per SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.
    1. The kernel uses 128 threads per block and 30 registers per thread.

        $2048 \div 128 = 16 \text{ blocks}$
        
        $2048 \text{ threads} \times 30 \text{ registers per thread} = 61440 \text{ registers}$

        The kernel can achieve full occupancy.

    2. The kernel uses 32 threads per block and 29 registers per thread.

         $2048 \div 32 = 64 \text{ blocks}$,

         How the maximum number of blocks is actually 32, it can use at most:
        $32 \text{ blocks} \times 32 \text{ threads per block} = 1024 \text{ threads}$

        $1024 \text{ threads } \times 29 \text{ registers per thread} = 29696 \text{ registers}$

        It can't reach full occupancy because it can't use all available threads.

    3. The kernel uses 256 threads per block and 34 registers per thread.

        $2048 \div 128 = 8 \text{ blocks}$

        $2048 \text{ threads} \times 34 \text{ registers per thread} = 69632 \text{ registers}$

        It can't reach full occupancy due the usage of registers, it can use at most $\lfloor 65,536 \div 34 \rfloor = 1927 \text{ threads}$ in a single SM.

9. A student mentions that they were able to multiply two $1024 \times 1024$ matrices using a matrix multiplication kernel with $32 \times 32$ thread blocks. The student is using a CUDA device that allows up to 512 threads per block and up to 8 blocks per SM. The student further mentions that each thread in a thread block calculates one element of the result matrix. What would be your reaction and why?

    I would think it was weired. First, $32 \times 32$ thread blocks results in 1024 threads in total, what is twice as much the limit of 512 threads per block. Also, if each thread produces one output, even if they were actually only using  the 512 threads per block and  all the 8 blocks, it would be only 4096 output elements per SM. It would require 256 SMs in the GPU. As of 2025, there is no GPU in history with that number os SMs as far as I know.
