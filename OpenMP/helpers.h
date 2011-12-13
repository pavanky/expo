#include <stdlib.h>
#include <time.h>
#include "timers.h"

#define TIME(_fn, _str, MAX_ITER) do {                              \
        for (int i = 0; i < 2; i++) _fn;                            \
        TIC;                                                        \
        for (int i = 0; i < MAX_ITER; i++) _fn;                     \
        TOC;                                                        \
        uSec /= MAX_ITER * 1000;                                    \
        printf("Time taken for %s: %4.4lf mSecs\n", _str, uSec);    \
    }while(0)

float* create_rand(int L) { float *M =  malloc(L * sizeof(float)); return M; }
float* create_zero(int L) { float *M =  malloc(L * sizeof(float)); return M; }

void destroy(float *M) { free(M); }
