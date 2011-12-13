#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "knn.h"

static int elsize = 0;
void generateData(pattern *S, pattern *T)
{
    pattern *A = S;
    pattern *B = S + CLASS_A;
    for (int i = 0; i < CLASS_A; i++) {
        A[i].value[0] = RAND() - 0.5;
        A[i].value[1] = RAND() - 0.8;
        A[i].cls = 1;
    }
    for (int i = 0; i < CLASS_B; i++) {
        B[i].value[0] = RAND() + 0.5;
        B[i].value[1] = RAND() + 0.2;
        B[i].cls = 2;
    }
    for (int i = 0; i < TEST_SET; i++) {
        T[i].value[0] = RAND();
        T[i].value[1] = RAND();
        T[i].cls = 0; // not set
    }
}

void nnCPU(pattern *S, pattern *T)
{
    for (int i = 0; i < TEST_SET; i++) {
        // Find nearest neighbor
        float minVal = dist(T[i], S[0]);
        int   minPos = 0;
        for (int j = 1; j < SAMPLE_SET; j++) {
            float val = dist(T[i], S[j]);
            if (val < minVal) {
                minVal = val;
                minPos = j;
            }
        }

        // Write the correct class
        T[i].cls = S[minPos].cls;
    }

}

template <bool is_batch>
void nnGPU(pattern *S, pattern *T)
{
    int threads = THREADS;
    int *P = NULL;
    float *V = NULL;
    if (!is_batch) {
        int blocks = DIVUP(SAMPLE_SET, (threads * REPEAT));
        int repeat = DIVUP(blocks, threads);
        CUDA_TRY(cudaMalloc(&P, blocks * sizeof(int)));
        CUDA_TRY(cudaMalloc(&V, blocks * sizeof(float)));
        for (int i = 0; i < TEST_SET; i++) {
            tmpMinDist<<<blocks, threads>>>(S, T, i, P, V);
            finMinDist<<<1 , threads>>>(P, V, T, S, i, repeat, blocks);
        }
    }
    else {
        int blocks_x =  DIVUP(SAMPLE_SET, (threads * REPEAT));
        int blocks_y = TEST_SET;
        int repeat = DIVUP(blocks_x, threads);
        CUDA_TRY(cudaMalloc(&P, blocks_x * blocks_y * sizeof(int)));
        CUDA_TRY(cudaMalloc(&V, blocks_x * blocks_y * sizeof(float)));
        dim3 blocks(blocks_x, blocks_y);
        tmpMinDist<<<blocks, threads>>>(S, T, -1, P, V); blocks.x = 1;
        finMinDist<<<blocks, threads>>>(P, V, T, S, -1, repeat, blocks_x);
    }
    CUDA_TRY(cudaFree(P));
    CUDA_TRY(cudaThreadSynchronize());
}

void testResults(pattern *T, pattern *gT)
{
    pattern TMP[TEST_SET];
    CUDA_TRY(cudaMemcpyD2H(TMP, gT, elsize * TEST_SET));
    for (int i = 0; i < TEST_SET; i++) {
        if(T[i].cls != TMP[i].cls) {
            printf("Error while testing results\n");
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    pattern *S = (pattern *)malloc(SAMPLE_SET * sizeof(pattern));
    pattern *T = (pattern *)malloc( TEST_SET  * sizeof(pattern));

    generateData(S, T);

    printf("Sample set: %d\n", SAMPLE_SET);
    printf("Test set: %d\n", TEST_SET);
    printf("Class a : %d, Class b : %d\n", CLASS_A, CLASS_B);

    // Alloc space on GPU
    pattern *gS, *gT, *gG;
    elsize = sizeof(pattern);
    CUDA_TRY(cudaMalloc(&gS, SAMPLE_SET * elsize));
    CUDA_TRY(cudaMalloc(&gT, TEST_SET   * elsize));
    CUDA_TRY(cudaMalloc(&gG, TEST_SET   * elsize));

    //Transfer data to gpu
    CUDA_TRY(cudaMemcpyH2D(gS, S, SAMPLE_SET * elsize));
    CUDA_TRY(cudaMemcpyH2D(gT, T, TEST_SET   * elsize));
    // To make sure the two methods write results to different locations
    CUDA_TRY(cudaMemcpyH2D(gG, T, TEST_SET   * elsize));

    // Nearest neighbor on cpu
    TIME(nnCPU(S, T), "CPU");

    // GPU using for. Slower but more space efficient
    TIME(nnGPU<false>(gS, gT), "GPU Method 1");

    // Test results
    testResults(T, gT);

    // GPU using batch. Faster but less space efficeint
    TIME(nnGPU<true>(gS, gG), "GPU Method 2");

    // Test results
    testResults(T, gG);

    // Free gpu memory
    CUDA_TRY(cudaFree(gS));
    CUDA_TRY(cudaFree(gT));
    CUDA_TRY(cudaFree(gG));

    // Free host memory
    free(S);
    free(T);
    return 0;
}
