#!/bin/sh

set -xe

clang -Wall -Wextra -o main ./src/main.c ./src/lin.c ./src/nn.c ./src/fun.c -lm
