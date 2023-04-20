const char *dgemv_desc = "Basic implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int size, double *A, double *x, double *y)
{
  // insert your code here: implementation of basic matrix multiply
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      y[i] += A[j + (i * size)] * x[j];
}

/**
 * P.S.
 *
 * I spent a total of 6 hours trying to figure out why the results I have in Y is different than the one from BLAS.
 * The way I implemented the code works in my head since that's how cross product works.
 * This haunted me in my sleep. I think my brain was like going through all possible fixes for this problem while I was asleep.
 * Then I realized on my way to class that the A matrix is basically a 1D array and not 2D.
 * The entire time, I was trying to implement the code based on the assumption that the A matrix was a 2D array.
 */