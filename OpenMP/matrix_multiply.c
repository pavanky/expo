#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
//#include <omp.h>

void naive1(int M, int N, float *A,
            int K, float *B, float *C, float *S)
{
	int k;
#if defined (_OPENMP)
#pragma omp parallel for
#endif
    for (k = 0; k < K; k++)
    {	register int m,n; register float tmp;
        for (m = 0; m < M; m++) {
            tmp = 0;
            for (n = 0; n < N; n++)
                tmp += A[m * N + n] * B[n * K + k];
            C[m * K + k] = tmp;
        }
    }
}

void naive2(int M, int N, float *A,
            int K, float *B, float *C, float *S)
{
	int m;
#if defined (_OPENMP)
#pragma omp parallel for
#endif
    for (m = 0; m < M; m++)
    {	register int k,n; register float tmp;
        for (k = 0; k < K; k++) {
            tmp = 0;
            for (n = 0; n < N; n++)
                tmp += A[m * N + n] * B[n * K + k];
            C[m * K + k] = tmp;
        }
    }
}

void naivec(int M, int N, float *A,
            int K, float *B, float *C, float *S)
{
    return (M > K) ? naive1(M, N, A, K, B, C, S) : naive2(M, N, A, K, B, C, S);
}

void trans(int M, int N, float *A,
           int K, float *B, float *C, float *S)
{
	int k;
#if defined (_OPENMP)
#pragma omp parallel for
#endif
    for (k = 0; k < K; k++) {
	    int n,m;
	    float tmp;
	    int k_x_N = k * N;
        for (n = 0; n < N; n++)
            S[k_x_N + n] = B[n * K + k];
        for (m = 0; m < M; m++) {
            tmp = 0;
            for (int n = 0; n < N; n++)
                tmp += A[m * N +n] * S[k_x_N + n];
            C[m * K + k] = tmp;
        }
    }

}

int main(int argc, char **args)
{
    float *A = NULL, *B = NULL, *C = NULL, *S = NULL;
    int c = 4;

    if (argc < c) {
        printf("USAGE %s <rowsA> <colsA> <colsB> <type [optional]>\n", args[0]);
        return -1;
    }

    int M = atoi(args[1]);
    int N = atoi(args[2]);
    int K = atoi(args[3]);

    printf("size of A: %d, %d\n", M, N);
    printf("size of B: %d, %d\n", N, K);
    printf("size of C: %d, %d\n", M, K);

    A = create_rand(M * N);
    B = create_rand(N * K);
    C = create_zero(M * K);
    S = create_zero((M+K) * N);

    do {
        int type = (argc > c) ? atoi(args[c]) : 0;
        switch (type) {
        case 1:
            TIME({ naive1(M, N, A, K, B, C, S); }, "naive1 matrix multiply", 5);
            break;
        case 2:
            TIME({ naive2(M, N, A, K, B, C, S); }, "naive2 matrix multiply", 5);
            break;
        case 3:
            TIME({ trans(M, N, A, K, B, C, S); }, "trans matrix multiply", 5);
            break;
        default:
            TIME({ naivec(M, N, A, K, B, C, S); }, "naivec matrix multiply", 5);
        }
    } while (++c != argc);

    destroy(A);
    destroy(B);
    destroy(C);
    destroy(S);
}
