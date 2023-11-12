#ifndef NN_H
#define NN_H

/*
    Simple neural network module for machine learning algorithms
*/

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

#include <stddef.h>

#include "lin.h"

typedef struct {
    size_t depth;
    Mat *W;
    Mat *b;
    Mat *a;
} NN;

#define NN_INPUT(nn) (nn).a[0]
#define NN_OUTPUT(nn) (nn).a[(nn).depth]

NN nn_init(size_t *layers, size_t depth);

NN nn_alloc(size_t *layers, size_t depth);

void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);

void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
void nn_mdiff(NN nn, Mat m, Mat g, Mat train_i, Mat train_o, float cost, float eps);
void nn_fdiff(NN nn, NN g, Mat train_i, Mat train_o, float eps);
void nn_learn(NN nn, NN g, float rate);

float nn_cost(NN nn, Mat train_i, Mat train_o);

// Get random float value
float rand_float(void);

// Sigmoid function
float sigmoidf(float x);

#endif
