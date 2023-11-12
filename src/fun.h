#ifndef FUN_H
#define FUN_H

/*
    A small function library with implementations for:

        - for_each
        - map
        - reduce
        - rand_float
        - sigmoidf
        
*/

#ifndef F_ASSERT
#include <assert.h>
#define F_ASSERT assert
#endif

#include <stddef.h>

void ffor_each(void (*f)(float), float *arr, size_t n);
void ifor_each(void (*f)(int),   int *arr,   size_t n);

#define fFOR_EACH(f, arr) ffor_each((f), (&arr), ARRAY_LEN(arr))
#define iFOR_EACH(f, arr) ifor_each((f), (&arr), ARRAY_LEN(arr))

void fmap(float (*f)(float), float *src, float *dst, size_t n, size_t m);
void imap(int   (*f)(int),   int *src,   int *dst,   size_t n, size_t m);

#define fMAP(f, src, dst) fmap((f), (&src), (&dst), ARRAY_LEN(src), ARRAY_LEN(dst))
#define iMAP(f, src, dst) imap((f), (&src), (&dst), ARRAY_LEN(src), ARRAY_LEN(dst))

float freduce(float (*f)(float, float), float *arr, size_t n);
int   ireduce(int   (*f)(int,   int),     int *arr, size_t n);

#define fREDUCE(f, arr) freduce((f), (&arr), ARRAY_LEN(arr))
#define iREDUCE(f, arr) ireduce((f), (&arr), ARRAY_LEN(arr))

// Get random float value
float rand_float(void);

// Sigmoid function
float sigmoidf(float x);

#endif