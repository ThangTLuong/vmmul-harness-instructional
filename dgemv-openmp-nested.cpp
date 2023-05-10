#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>

const char *dgemv_desc = "OpenMP dgemv nested.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */

void my_dgemv(int size, double *A, double *x, double *y)
{

  // #pragma omp parallel
  // {
  // int nthreads = omp_get_num_threads();
  // int thread_id = omp_get_thread_num();
  // printf("Hello world: thread %d of %d checking in. \n", thread_id, nthreads);
  // }

#pragma omp parallel for
  for (int i = 0; i < size; i++)
  {
    double temp = 0.0;
#pragma omp parallel for reduction(+ : temp)
    for (int j = 0; j < size; j++)
      temp += A[j + (i * size)] * x[j];
    y[i] += temp;
  }

  // insert your dgemv code here. you may need to create additional parallel regions,
  // and you may want to comment out the above parallel code block that prints out
  // nthreads and thread_id so as to not taint your timings
}
