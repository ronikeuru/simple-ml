#!/bin/sh

set -xe

# clang -Wall -Wextra -o twice twice.c functions.c -lm
# clang -Wall -Wextra -o gates gates.c functions.c -lm
# clang -Wall -Wextra -o main main.c functions.c xor.c -lm

clang -Wall -Wextra -o main ./src/main.c ./src/lin.c ./src/nn.c -lm