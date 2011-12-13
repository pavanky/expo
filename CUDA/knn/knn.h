#include "cubasics.h"

#define SAMPLE_SET 5000
#define CLASS_A (SAMPLE_SET / 2)
#define CLASS_B (SAMPLE_SET / 2)
#define TEST_SET 5000
#define MAX_ITER 5
#define REPEAT 32

#define TIME(fn, str) do {                                              \
        clock_t start = clock();                                        \
        for (int iter = 0; iter < MAX_ITER; iter++) fn;                 \
        clock_t end = clock();                                          \
        float msecs = 1000 * (end - start) / CLOCKS_PER_SEC;            \
        printf("Time taken for %s: %4.4f ms\n", str, msecs/MAX_ITER);   \
    }while(0)

#define RAND() (((float)rand())/RAND_MAX)

typedef struct {
    float value[2];
    int cls;
} pattern;

// function used both on host (cpu) and device (gpu)
__device__ __host__ float dist(pattern a, pattern b)
{
    float df0 = a.value[0] - b.value[0];
    float df1 = a.value[1] - b.value[1];
    return (df0*df0 + df1*df1);
}

__device__ void getMin(float val, int i, float *lVal, int *lPos)
{
    if (val < *lVal && i != -1) {
        *lVal = val;
        *lPos = i;
    }
}

static __global__ void tmpMinDist(pattern *S, pattern *T, int n,
                                  int *P, float *V)
{

    // SAMPLE_SET data parallel along X dim
    // Each thread handles REPEAT number from SAMPLE_SET
    // there are blockDim.x number of threads per block
    // so the starting sample of each thread is at the following location
    int start = blockIdx.x * REPEAT * blockDim.x + threadIdx.x;
    int end = min(start + REPEAT * blockDim.x, SAMPLE_SET);

    // if using batch, TEST_SET parallel along y dim
    if (n < 0) {
        n = blockIdx.y; // so n is the id of block along y dim
        P += n * gridDim.x; // Offset the result array to correct location
        V += n * gridDim.x;
    }

    // min values in shared memory
    __shared__ float sVal[THREADS];
    __shared__ int   sPos[THREADS];

    float *lVal = sVal + threadIdx.x;
    int   *lPos = sPos + threadIdx.x;

    *lPos = -1;
    if (start >= end) return;

    int i = start;
    *lVal = dist(T[n], S[i]);
    *lPos = start;
    i += blockDim.x;
    for (; i < end; i += blockDim.x) {
        float cVal = dist(T[n], S[i]);
        getMin(cVal, i, lVal, lPos);
    }
    __syncthreads(); // make sure all threads in the block are here

    int tid = threadIdx.x;
    if (tid >= 128) return;
    getMin(lVal[128], lPos[128], lVal, lPos); __syncthreads();
    if (tid >=  64) return;
    getMin(lVal[ 64], lPos[ 64], lVal, lPos); __syncthreads();
    if (tid >=  32) return;
    getMin(lVal[ 32], lPos[ 32], lVal, lPos); __syncthreads();

    // intra warp, __syncthreads() not reqd
    if (tid < 16) getMin(lVal[ 16], lPos[ 16], lVal, lPos);
    if (tid <  8) getMin(lVal[  8], lPos[  8], lVal, lPos);
    if (tid <  4) getMin(lVal[  4], lPos[  4], lVal, lPos);
    if (tid <  2) getMin(lVal[  2], lPos[  2], lVal, lPos);
    if (tid <  1) getMin(lVal[  1], lPos[  1], lVal, lPos);

    // write out the results
    if (tid == 0) {
        V[blockIdx.x] = *lVal;
        P[blockIdx.x] = *lPos;
    }
}

static __global__ void finMinDist(int *P, float *V, pattern *T, pattern *S,
                                  int n, int repeat, int blocks)
{
    int tid = threadIdx.x;
    int start = threadIdx.x;
    int end = min(start + repeat * blockDim.x, blocks);

    if (n < 0) {
        n = blockIdx.y; // so n is the id of block along y dim
        P += n * blocks;
        V += n * blocks;
    }

    __shared__ float sVal[THREADS];
    __shared__ int   sPos[THREADS];

    float *lVal = sVal + threadIdx.x;
    int   *lPos = sPos + threadIdx.x;

    *lPos = -1;
    if (start >= end) return;

    int i = start;
    *lVal = V[i];
    *lPos = P[i];
    i += blockDim.x;
    for (; i < end; i += blockDim.x) {
        float cVal = V[i];
        int j = P[i];
        getMin(cVal, j, lVal, lPos);
    }
    __syncthreads(); // make sure all threads in the block are here

    if (tid >= 128) return;
    getMin(lVal[128], lPos[128], lVal, lPos); __syncthreads();
    if (tid >=  64) return;
    getMin(lVal[ 64], lPos[ 64], lVal, lPos); __syncthreads();
    if (tid >=  32) return;
    getMin(lVal[ 32], lPos[ 32], lVal, lPos); __syncthreads();

    // intra warp, __syncthreads() not reqd
    if (tid < 16) getMin(lVal[ 16], lPos[ 16], lVal, lPos);
    if (tid <  8) getMin(lVal[  8], lPos[  8], lVal, lPos);
    if (tid <  4) getMin(lVal[  4], lPos[  4], lVal, lPos);
    if (tid <  2) getMin(lVal[  2], lPos[  2], lVal, lPos);
    if (tid <  1) getMin(lVal[  1], lPos[  1], lVal, lPos);

    // write out the results
    if (tid == 0) T[n].cls = S[*lPos].cls;
}
