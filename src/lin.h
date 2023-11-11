#ifndef LIN_H
#define LIN_H

/*
    Simple linear algebra module for machine learning algorithms
*/

#include <stddef.h>

#ifndef MAT_MALLOC
#include <stdlib.h>
#define MAT_MALLOC malloc
#endif

#ifndef MAT_ASSERT
#include <assert.h>
#define MAT_ASSERT assert
#endif

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

// Get matrix with allocated memory
Mat mat_alloc(size_t rows, size_t cols);

// Get a given row from a matrix
Mat mat_row(Mat m, size_t row);
// Get a given column from a matrix
Mat mat_col(Mat m, size_t col);

// Copy content of matrix to another
void mat_copy(Mat dst, Mat src);
// Fill a matrix with given values
void mat_fill(Mat m, float x);
// Initialize a matrix with random values from given interval
void mat_rand(Mat m, float low, float high);
// Dot product for given matrices: a * b = c
void mat_dot(Mat a, Mat b, Mat c);
// Sum of two matrices: a + b = a
void mat_sum(Mat a, Mat b);
// Apply sigmoid function for the values of a matrix
void mat_sig(Mat m);
// Apply gradient matrix to model matrix with certain rate
void mat_learn(Mat m, Mat g, float rate);
// Print the content of a matrix
void mat_print(Mat m, const char *name, size_t padding);

#define MAT_PRINT(m) mat_print(m, #m, 0);

#endif