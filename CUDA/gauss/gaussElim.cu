#include<stdio.h>
#define DEBUG 0
#define N DEBUG ? 3:1000
#define M DEBUG ? 3:1500
#define type float
#define THREADS 256
#define MAXS 2048
#define SAFE(A) do{                                                     \
        cudaError_t e = A;                                              \
        if (e) {                                                        \
            printf("exited CUDA error: %s\n", cudaGetErrorString(e));    \
            return -1;                                                  \
        }                                                               \
    }while(0)
void read_Data(type *A, type *B, int n, int m)
{
#if DEBUG
    type I[] = {10, 10, 1, 2, 5, 4, 10, 8, 9};
    for (int i = 0; i < n * n; i++) A[i] = I[i];
    for (int i = 0; i < n * m; i++) B[i] = (i + 1);
#endif
    //Use customized read functions here

}

__global__ void eliminate(type *A, type *B, int x, int n, int m)
{
    int row = blockIdx.x;
    int tx = threadIdx.x;
    int ts = blockDim.x;
    __shared__ type sfactor;
    int i = tx;
    type factor;
    int off = blockIdx.y * ts;
    int init = off + x   * n; // row corresponding to the pivot
    int curr = off + row * n; // row that you are operating on
    if (row == x) return;

    //read pivots and calculate multiply factor only once
    if (tx == 0) sfactor = A[curr + x] / A[init + x];
    __syncthreads();

    // make a local copies of factor
    factor = sfactor;

    // do the actual calculations
    if ((i >= x) && (off + i < n))// everything ahead is 0
            A[curr + i] -= factor * A[init + i];

    init = off + x   * m; // row corresponding to the pivot
    curr = off + row * m; // row that you are operating on
    if (off + i < m) B[curr + i] -= factor * B[init + i];
}

__global__ void normalize(type *A, type *B, int n, int m)
{
    int row = blockIdx.x;
    int ts = blockDim.x;
    int tx = threadIdx.x;
    __shared__ type sfactor;
    type factor;

    // offset to current row
    int off = blockIdx.y * ts;
    int curr = off + row * n;
    // get normalizing factor
    if (tx == 0) sfactor = A[curr + row];
    __syncthreads();
    // get local copy
    factor = sfactor;
    int i = tx;

    curr = row * m; // row that you are operating on
    // normalize
    if (off + i < m) B[curr + i] /= factor;
}

int main(int argc, char *argv[])
{
    int n = N; // replace with appropriate value
    int m = M; // replace with appropriate value
    int bytesA = n * n * sizeof(type);
    int bytesB = n * m * sizeof(type);

    // allocate data on host
    type *h_A = (type *)malloc(bytesA);
    type *h_B = (type *)malloc(bytesB);

    // read data
    read_Data(h_A, h_B, n, m);

    // allocate data on device
    type *d_A; SAFE(cudaMalloc(&d_A, bytesA));
    type *d_B; SAFE(cudaMalloc(&d_B, bytesB));

    // transfer data
    SAFE(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    SAFE(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    int threads = THREADS;
    dim3 blocks(n, (n + threads - 1)/n);

    for (int i = 0; i < n; i++)
             eliminate<<<blocks, threads>>>(d_A, d_B, i, n, m);
    normalize<<<blocks, threads>>>(d_A, d_B, n, m);

#if DEBUG
    SAFE(cudaMemcpy(h_A, d_A, bytesA, cudaMemcpyDeviceToHost));
    SAFE(cudaMemcpy(h_B, d_B, bytesB, cudaMemcpyDeviceToHost));
    for (int j = 0; j < n; j++) { for (int i = 0; i < n; i++) printf("%.4f ", h_A[j*n + i]); printf("\n"); }
    for (int j = 0; j < n; j++) { for (int i = 0; i < m; i++) printf("%.4f ", h_B[j*m + i]); printf("\n"); }
#endif
}
