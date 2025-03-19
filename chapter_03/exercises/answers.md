# Exercises chapter 3

3. Consider the following CUDA kernel and the corresponding host function that calls it:
    ```c
    // can't past that code :`)
    ```
    1. What is the number of threads per block? 
    
        $16 \times 32 = 512$
    
    2. What is the number of threads in the grid?
        
        The grid is $19 \times 5$

        $19 \times 5 \times 512 = 48,640$
    
    3. What is the number of blocks in the grid?

        $19 \times 5 = 95$
    
    4. What is the number of threads that execute the code on line 05?

        $150 \times 300 = 45,000$ threads

4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of the matrix element at row 20 and column 10: 
    
    1. If the matrix is stored in row-major order.

        $20\times400 + 10 = 8010$
    
    2. If the matrix is stored in column-major order.

        $10\times500 + 20 = 5020$

5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x = 10, y = 20, and z = 5;

    $x\times\text{WIDTH} + y + z(\text{WIDTH}\times\text{HEIGHT})$

    $(10\times400 + 20) + (400\times500\times5) = 1004020$
