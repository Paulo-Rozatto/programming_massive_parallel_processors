# Exercises chapter 7

1. Calculate the P[0] value in Fig. 7.3.
    - **Filter**: {1, 3, 5, 3, 1}
    - **X**: {8 , 2, 5, 4, 1, 7, 3}
    - $P[0] = (1 \times 0) + (3 \times 0) + (5 \times 8) + (3 \times 2) + (1 \times 5) = 51 $

2. Consider performing a 1D convolution on array N = {4, 1, 3, 2, 3} with filter F = {2, 1, 4}. What is the resulting output array?

    Considering that we will pad the boundaries with zeros:
    - $P[0] = (2 \times 0) + (1 \times 4) + (4 \times 1) = 8 $
    - $P[1] = (2 \times 4) + (1 \times 1) + (4 \times 3) = 21 $
    - $P[2] = (2 \times 1) + (1 \times 3) + (4 \times 2) = 13 $
    - $P[3] = (2 \times 3) + (1 \times 2) + (4 \times 3) = 20 $
    - $P[4] = (2 \times 2) + (1 \times 3) + (4 \times 0) = 7 $

3. What do you think the following 1D convolution filters are doing?
    1. [0 1 0]

            Identity

    2. [0 0 1]

            Shift Left

    3. [1 0 0]

            Shift Right

    4. [$-\frac{1}{2}$ 0 $\frac{1}{2}$]

            High-pass filter

    5. [$\frac{1}{3}$ $\frac{1}{3}$ $\frac{1}{3}$]

            Low-pass filter (averaging)

4. Consider performing a 1D convolution on an array of size N with a filter of size M:
    1. How many ghost cells are there in total?

        There will be $M - 1$ ghost cells padding the two borders.
        
        That means that there will be $M - 1$ multiplications by ghost cells considering the last elements in both sides, $M - 2$ multiplications by ghost cells for the second last, and so on until $M - M = 0$. That is an arithimetic progression:

        $$\frac{0 + M - 1}{2} \times \frac{M + 1}{2} = \frac{M^2 - 1}{4}$$

    2. How many multiplications are performed if ghost cells are treated as multiplications (by 0)?

        $N \times M$

    3. How many multiplications are performed if ghost celss are not treated as multiplications?
        
        - If they are just ignored: 
        
            $N \times M - \frac{M^2 - 1}{4}$
        - If no filter with ghost cells is computed: 

            $M \times (N - M + 1)$

5. Consider performing a 2D convolution on a square matrix of size $N \times N$ with a square filter of size $M \times M$:
    1. How many ghost cells are there in total?
        
        Each side will be padded with $N \times \frac{M - 1}{2}$ ghost cells and there will be $(\frac{M - 1}{2})^2$ on the corners.

    2. How many multiplications are performed if ghost cells are treated as multiplications (by 0)?

        $N^2 \times M^2$

    3. How many multiplications are performed if ghost cells are not treated as multiplications?
        
        - If no filter with ghost cells is computed: 
            
            $M^2 \times (N - M + 1)^2$