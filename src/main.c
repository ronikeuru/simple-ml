#include <time.h>
#include <stdio.h>
#include <math.h>

#include "nn.h"

float xor_train_data[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

float or_train_data[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

int main(void)
{
    srand(time(0));

    float *train_data = xor_train_data;

    size_t stride = 3;
    size_t n = 4;
    
    Mat train_i = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = train_data,
    };

    Mat train_o = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = train_data + 2,
    };

    size_t layers[] = {2, 2, 1};
    NN nn = nn_init(layers, ARRAY_LEN(layers));
    NN g  = nn_init(layers, ARRAY_LEN(layers));

    float eps = 1e-1;
    float rate = 1e-1;

    float cost = nn_cost(nn, train_i, train_o);
    printf("cost = %f\n", cost);
    while (cost > 0.1f) {
        // Calculate gradient
        nn_fdiff(nn, g, train_i, train_o, eps);

        // Apply gradient for learning
        nn_learn(nn, g, rate);
        
        cost = nn_cost(nn, train_i, train_o);
    }
    printf("cost = %f\n", cost);

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            MAT_AT(NN_INPUT(nn), 0, 0) = i;
            MAT_AT(NN_INPUT(nn), 0, 1) = j;

            nn_forward(nn);

            printf("%zu ^ %zu = %f\n", i, j, roundf(MAT_AT(NN_OUTPUT(nn), 0, 0)));
        }
    }

    return 0;
}