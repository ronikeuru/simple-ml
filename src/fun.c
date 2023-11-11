#include <stdlib.h>
#include <math.h>

#include "fun.h"

void ffor_each(void (*f)(float), float *arr, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        f(arr[i]);
    }
}

void ifor_each(void (*f)(int), int *arr, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        f(arr[i]);
    }
}

void fmap(float (*f)(float), float *src, float *dst, size_t n, size_t m)
{
    F_ASSERT(n == m);
    
    for (size_t i = 0; i < n; i++) {
        dst[i] = f(src[i]);
    }
}

void imap(int (*f)(int), int *src, int *dst, size_t n, size_t m)
{
    F_ASSERT(n == m);
    
    for (size_t i = 0; i < n; i++) {
        dst[i] = f(src[i]);
    }
}


float freduce(float (*f)(float, float), float *arr, size_t n)
{
    float y = 0.f;

    for (size_t i = 0; i < n; i++) {
        y = f(y, arr[i]);
    }

    return y;
}


int ireduce(int (*f)(int, int), int *arr, size_t n)
{
    int y = 0;
    
    for (size_t i = 0; i < n; i++) {
        y = f(y, arr[i]);
    }

    return y;
}

// Get random float value
float rand_float(void)
{
    return (float) rand() / (float) RAND_MAX;
}

// Sigmoid function
float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}
