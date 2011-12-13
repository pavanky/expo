#ifndef __CUBASICS__
#define __CUBASICS__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define MAXBLKS 65535
#define MINBLKS 30
#define THREADS 256
#define XDIM 32
#define YDIM  8

#define cudaMemcpyH2D(d,s,b) cudaMemcpy(d,s,b,cudaMemcpyHostToDevice)
#define cudaMemcpyD2H(d,s,b) cudaMemcpy(d,s,b,cudaMemcpyDeviceToHost)

#define MSG(msg,...) do {                               \
        fprintf(stdout,__FILE__":%d(%s) " msg "\n",     \
                __LINE__, __FUNCTION__, ##__VA_ARGS__); \
        fflush(stdout);                                 \
    } while (0)

#define CUDA_TRY(fn) do {                                   \
        cudaError_t e = fn;                                 \
        if (e != cudaSuccess) {                             \
            MSG("CUDA error : %s",cudaGetErrorString(e));   \
            exit(3);                                        \
        }                                                   \
    }while(0)

#define DIVUP(a, b) ((a+b)/b)

#endif
