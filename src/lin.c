#include "lin.h"
#include "fun.h"


#include <stdio.h>

// Get matrix with allocated memory
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = malloc(sizeof(*m.es)*rows*cols);

    assert(m.es != NULL);
    
    return m;
}

// Get a given row from a matrix
Mat mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,
        .es = &MAT_AT(m, row, 0),
    };
}

// Get a given column from a matrix
Mat mat_col(Mat m, size_t col)
{
    return (Mat) {
        .rows = m.rows,
        .cols = 1,
        .stride = 1,
        .es = &MAT_AT(m, 0, col),
    };
}

// Copy content of matrix to another
void mat_copy(Mat dst, Mat src)
{
    // Row and column dimensions have to match
    MAT_ASSERT(dst.rows == src.rows);
    MAT_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

// Fill a matrix with given values
void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = x;
        }
    }
}

// Initialize a matrix with random values from given interval
void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_float()*(high - low) + low;
        }
    }
}

// Dot product for given matrices: a * b = c
void mat_dot(Mat a, Mat b, Mat c)
{
    // Column dimension of a and row dimension of b have to match
    MAT_ASSERT(a.cols == b.rows);
    // Row dimensions of a and c have to match
    MAT_ASSERT(a.rows == c.rows);
    // Column dimensions of b and c have to match
    MAT_ASSERT(b.cols == c.cols);
    // Reason:
    // a_(m x n) * b_(n x k) = c_(m x k)

    size_t n = a.cols;

    for (size_t i = 0; i < c.rows; i++) {
        for (size_t j = 0; j < c.cols; j++) {
            MAT_AT(c, i, j) = 0;
            for (size_t k = 0; k < n; k++) {
                MAT_AT(c, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

// Sum of two matrices: a + b = a
void mat_sum(Mat a, Mat b)
{
    // Row and column dimensions of a and b have to match
    MAT_ASSERT(a.rows == b.rows);
    MAT_ASSERT(a.cols == b.cols);

    for (size_t i = 0; i < a.rows; i++) {
        for (size_t j = 0; j < a.cols; j++) {
            MAT_AT(a, i, j) += MAT_AT(b, i, j);
        }
    }
}

// Apply sigmoid function for the values of a matrix
void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

// Apply gradient matrix to model matrix with certain rate
void mat_learn(Mat m, Mat g, float rate)
{
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
                MAT_AT(m, i, j) -= rate * MAT_AT(g, i, j);
        }
    }
}

// Print the content of a matrix
void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; i++) {
        printf("%*s   ", (int) padding, "");
        for (size_t j = 0; j < m.cols; j++) {
            printf("%f  ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}