#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "nn.h"
#include "fun.h"

NN nn_alloc(size_t *layers, size_t depth)
{
    NN nn;

    nn.depth = depth - 1;

    nn.W = NN_MALLOC(sizeof(*nn.W)*nn.depth);
    NN_ASSERT(nn.W != NULL);
    nn.b = NN_MALLOC(sizeof(*nn.b)*(nn.depth));
    NN_ASSERT(nn.b != NULL);
    nn.a = NN_MALLOC(sizeof(*nn.a)*(nn.depth + 1));
    NN_ASSERT(nn.a != NULL);

    nn.a[0] = mat_alloc(1, layers[0]);
    for (size_t i = 1; i < depth; ++i) {
        nn.W[i-1] = mat_alloc(nn.a[i-1].cols, layers[i]);
        nn.b[i-1] = mat_alloc(1, layers[i]);
        nn.a[i]   = mat_alloc(1, layers[i]);
    }

    return nn;
}

NN nn_init(size_t *layers, size_t depth)
{
    NN_ASSERT(depth > 0);

    NN nn = nn_alloc(layers, depth);
    nn_rand(nn, 0, 1);
    
    return nn;
}

void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.depth; i++) {
        snprintf(buf, sizeof(buf), "W_%zu", i);
        mat_print(nn.W[i], buf, 4);
        snprintf(buf, sizeof(buf), "b_%zu", i);
        mat_print(nn.b[i], buf, 4);
    }
    printf("]\n");
}

void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.depth; i++) {
        mat_fill(nn.W[i], 0);
        mat_fill(nn.b[i], 0);
        mat_fill(nn.a[i], 0);
    }
    mat_fill(nn.a[nn.depth], 0);
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.depth; i++) {
        mat_rand(nn.W[i], low, high);
        mat_rand(nn.b[i], low, high);
    }
}

void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.depth; i++) {
        mat_dot(nn.a[i], nn.W[i], nn.a[i + 1]);
        mat_sum(nn.a[i + 1], nn.b[i]);
        mat_sig(nn.a[i + 1]);
    }
}

void nn_mdiff(NN nn, Mat m, Mat g, Mat train_i, Mat train_o, float cost, float eps)
{
    float saved;

    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            saved = MAT_AT(m, i, j);
            MAT_AT(m, i, j) += eps;
            MAT_AT(g, i, j) = (nn_cost(nn, train_i, train_o) - cost) / eps;
            MAT_AT(m, i, j) = saved;
        }
    }
}

void nn_fdiff(NN nn, NN g, Mat train_i, Mat train_o, float eps)
{
    float cost = nn_cost(nn, train_i, train_o);

    for (size_t i = 0; i < nn.depth; i++) {
        nn_mdiff(nn, nn.W[i], g.W[i], train_i, train_o, cost, eps);
    }
    for (size_t i = 0; i < nn.depth; i++) {
        nn_mdiff(nn, nn.b[i], g.b[i], train_i, train_o, cost, eps);
    }
}

void nn_backprop(NN nn, NN g, Mat train_i, Mat train_o)
{
    NN_ASSERT(train_i.rows == train_o.rows);
    size_t n = train_i.rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == train_o.cols);

    // Initialize the gradient
    nn_zero(g);

    for (size_t i = 0; i < n; ++i) {
        // Copy training input into nn intput layer
        mat_copy(NN_INPUT(nn), mat_row(train_i, i));

        // Run forward pass
        nn_forward(nn);

        // Clean up old gradient activations
        for (size_t j = 0; j <= nn.depth; j++) {
            mat_fill(g.a[j], 0);
        }

        // Calculate difference (error) with nn output and training output
        for (size_t j = 0; j < train_o.cols; j++) {
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(train_o, i, j);
        }

        // Run backwards through the nn layers
        for (size_t l = nn.depth; l > 0; l--) {
            // Go through the current activation output (a) columns
            for (size_t j = 0; j < nn.a[l].cols; j++) {
                // The activation output at current layer
                float a = MAT_AT(nn.a[l], 0, j);
                // The difference for the activation output
                float da = MAT_AT(g.a[l], 0, j);

                // Partial derivative for bias
                MAT_AT(g.b[l-1], 0, j) += 2*da*a*(1 - a);

                // Go through the previous activation output columns
                for (size_t k = 0; k < nn.a[l-1].cols; k++) {
                    // Previous activation
                    float pa = MAT_AT(nn.a[l-1], 0, k);
                    // Partial derivative for previous weights
                    MAT_AT(g.W[l-1], k, j) += 2*da*a*(1 - a)*pa;

                    // Previous weights
                    float w = MAT_AT(nn.W[l-1], k, j);
                    // Partial derivative for previous activation output
                    MAT_AT(g.a[l-1], 0, k) += 2*da*a*(1 - a)*w;
                }
            }
        }
    }

    // Divide everything with n
    for (size_t i = 0; i < g.depth; i++) {
        // Go through all the weights
        for (size_t j = 0; j < g.W[i].rows; j++) {
            for (size_t k = 0; k < g.W[i].cols; k++) {
                MAT_AT(g.W[i], j, k) /= n;
            }
        }

        // Go through all the biases
        for (size_t j = 0; j < g.b[i].rows; j++) {
            for (size_t k = 0; k < g.b[i].cols; k++) {
                MAT_AT(g.b[i], j, k) /= n;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.depth; i++) {
        mat_learn(nn.W[i], g.W[i], rate);
        mat_learn(nn.b[i], g.b[i], rate);
    }
}

float nn_cost(NN nn, Mat train_i, Mat train_o)
{
    // Training input and output row dimensions have to match
    NN_ASSERT(train_i.rows == train_o.rows);

    // Training and model output column dimensions have to match
    NN_ASSERT(train_o.cols == NN_OUTPUT(nn).cols);

    float sq_error = 0.f;

    size_t n = train_i.rows;
    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(train_i, i);
        Mat y = mat_row(train_o, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t m = train_o.cols;
        for (size_t j = 0; j < m; j++) {
            float diff = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            sq_error += diff*diff;
        }
    }

    return sq_error / n;
}
