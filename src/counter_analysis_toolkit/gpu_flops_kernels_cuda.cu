#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <papi.h>
#include "gpu_flops_kernels.h"
#include "gpu_flops.h"

#if defined(GPU_NVIDIA)

#define NUM_SIZES 3
#define NUM_TYPES 6
#define MAX_SIZE  64

static void gpu_matrix_flop(int EventSet, FILE *ofp_papi);

/* Wrapper functions of different precisions. */
#if defined(CAT_GPU_PREC_DP)
    extern "C" void gpu_matrix_flop_dp(int EventSet, FILE *ofp_papi) {
        gpu_matrix_flop(EventSet, ofp_papi);
    }
#elif defined(CAT_GPU_PREC_SP)
    extern "C" void gpu_matrix_flop_sp(int EventSet, FILE *ofp_papi) {
        gpu_matrix_flop(EventSet, ofp_papi);
    }
#elif defined(CAT_GPU_PREC_HP)
    extern "C" void gpu_matrix_flop_hp(int EventSet, FILE *ofp_papi) {
        gpu_matrix_flop(EventSet, ofp_papi);
    }
#endif

static void gpu_matrix_flop(int EventSet, FILE *ofp_papi) {

    int i, j, retval;
    CAT_GPU_PREC junk = 0.0;
    long long values = 0;
    size_t maxSize;
    size_t sizes[NUM_SIZES];
    int sizeIdx, typeIdx;
    int types[NUM_TYPES];
    cudaError_t status;

    CAT_GPU_PREC  *hostA,  *hostB,  *hostC;
    CAT_GPU_PREC   *devA,   *devB,   *devC;

    /* Create a list of types. */
    types[0] = ADD;
    types[1] = MUL;
    types[2] = DIV;
    types[3] = SQRT;
    types[4] = SUB;
    types[5] = FMA;

    /* Create a list of sizes. */
    sizes[0] = 16;
    sizes[1] = 32;
    sizes[2] = 64;

    maxSize = MAX_SIZE*MAX_SIZE*sizeof(CAT_GPU_PREC);

    /* Allocate host arrays. */
    status = cudaMallocHost((void**)&hostA, maxSize);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix A.\n");
        return;
    }
    status = cudaMallocHost((void**)&hostB, maxSize);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix B.\n");
        return;
    }
    status = cudaMallocHost((void**)&hostC, maxSize);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix C.\n");
        return;
    }

    /* Allocate device arrays. */
    status = cudaMalloc((void**)&devA, maxSize);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not allocate device matrix A.\n");
        return;
    }
    status = cudaMalloc((void**)&devB, maxSize);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not allocate device matrix B.\n");
        return;
    }
    status = cudaMalloc((void**)&devC, maxSize);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not allocate device matrix C.\n");
        return;
    }

    /* Randomly initialize arrays. */
    srandom(1);
    for( i = 0; i < MAX_SIZE; i++ ) {
        for( j = 0; j < MAX_SIZE; j++ ) {
            hostA[i*MAX_SIZE + j] = ((CAT_GPU_PREC)random())/((CAT_GPU_PREC)RAND_MAX) + (CAT_GPU_PREC)1.1;
            hostB[i*MAX_SIZE + j] = ((CAT_GPU_PREC)random())/((CAT_GPU_PREC)RAND_MAX) + (CAT_GPU_PREC)1.1;
            hostC[i*MAX_SIZE + j] = 0.0;
        }
    }

    /* Copy host data to device. */
    status = cudaMemcpy(devA, hostA, maxSize, cudaMemcpyHostToDevice);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not copy matrix A to device.\n");
        return;
    }
    status = cudaMemcpy(devB, hostB, maxSize, cudaMemcpyHostToDevice);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not copy matrix B to device.\n");
        return;
    }
    status = cudaMemcpy(devC, hostC, maxSize, cudaMemcpyHostToDevice);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not copy matrix C to device.\n");
        return;
    }

    /* Loop through the kernels. */
    for(typeIdx = 0; typeIdx < NUM_TYPES; ++typeIdx) {

        for(sizeIdx = 0; sizeIdx < NUM_SIZES; ++sizeIdx) {

            size_t N = sizes[sizeIdx];
            int type = types[typeIdx];

            /* Device configuration. */
            dim3 threads_per_block( 1, 1, 1 );
            dim3 blocks_in_grid( ceil( float(N) / threads_per_block.x ), ceil( float(N) / threads_per_block.y ), 1 );

            /* Start PAPI counters. */
            if( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
                fprintf(stderr, "GPU FLOPs Benchmark: PAPI_start() returned error code %d\n", retval);
                return;
            }

            /* Various floating-point operation kernels. */
            switch(type) {
              case ADD:
                matrix_add<<<blocks_in_grid, threads_per_block>>>(devA, devB, devC, N);
                break;
              case MUL:
                matrix_mul<<<blocks_in_grid, threads_per_block>>>(devA, devB, devC, N);
                break;
              case DIV:
                matrix_div<<<blocks_in_grid, threads_per_block>>>(devA, devB, devC, N);
                break;
              case SQRT:
                matrix_sqrt<<<blocks_in_grid, threads_per_block>>>(devA, devB, devC, N);
                break;
              case SUB:
                matrix_sub<<<blocks_in_grid, threads_per_block>>>(devA, devB, devC, N);
                break;
              case FMA:
                matrix_fma<<<blocks_in_grid, threads_per_block>>>(devA, devB, devC, N);
                break;
              default:
                break;
            }

            /* Error checking -- consider calling cudaGetLastError() after PAPI_stop(). */
            /*status = cudaGetLastError();
            if( cudaSuccess != status ) {
                fprintf(stderr, "Error 1.\n");
                return;
            }*/
            status = cudaDeviceSynchronize();
            if( cudaSuccess != status ) {
                fprintf(stderr, "Error 2.\n");
                return;
            }

            /* Stop PAPI counters. */
            if( (retval = PAPI_stop(EventSet, &values)) != PAPI_OK ) {
                fprintf(stderr, "GPU FLOPs Benchmark: PAPI_stop() returned error code %d\n", retval);
                return;
            }
            fprintf(ofp_papi, "%lld\n", values);

        }
    }

    /* Copy device data to host. */
    status = cudaMemcpy(hostC, devC, maxSize, cudaMemcpyDeviceToHost);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not copy matrix C from device.\n");
        return;
    }

    /* Use the result from the kernels to prevent compiler optimizing it away. */
    junk = ((CAT_GPU_PREC)1.23+hostC[MAX_SIZE*MAX_SIZE/2])/((CAT_GPU_PREC)1.45+hostC[4*MAX_SIZE*MAX_SIZE/5]*hostC[MAX_SIZE*MAX_SIZE-1]);
    if( junk > (CAT_GPU_PREC)1.23 && junk < (CAT_GPU_PREC)1.2345 )
        fprintf(stdout, "Benchmark artifact (%f) -- ignore.\n", (float)junk);

    /* Free device memory. */
    status = cudaFree(devA);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not free device matrix A.\n");
        return;
    }
    status = cudaFree(devB);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not free device matrix B.\n");
        return;
    }
    status = cudaFree(devC);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not free device matrix C.\n");
        return;
    }

    /* Free host memory. */
    status = cudaFreeHost(hostA);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not free host matrix A.\n");
        return;
    }
    status = cudaFreeHost(hostB);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not free host matrix B.\n");
        return;
    }
    status = cudaFreeHost(hostC);
    if( cudaSuccess != status ) {
        fprintf(stderr, "Could not free host matrix C.\n");
        return;
    }

    return;
}

__global__ void matrix_add(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) { /* This check shouldn't be necessary for now. */
        C[rowIdx*N + colIdx] = ADD_INTRIN(A[rowIdx*N + colIdx], B[rowIdx*N + colIdx]);
    }
}

__global__ void matrix_sub(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        C[rowIdx*N + colIdx] = SUB_INTRIN(A[rowIdx*N + colIdx], B[rowIdx*N + colIdx]);
    }
}

__global__ void matrix_mul(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        C[rowIdx*N + colIdx] = MUL_INTRIN(A[rowIdx*N + colIdx], B[rowIdx*N + colIdx]);
    }
}

__global__ void matrix_div(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        C[rowIdx*N + colIdx] = DIV_INTRIN(A[rowIdx*N + colIdx], B[rowIdx*N + colIdx]);
    }
}

__global__ void matrix_sqrt(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        C[rowIdx*N + colIdx] = SQRT_INTRIN(A[rowIdx*N + colIdx]);
    }
}

__global__ void matrix_fma(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {
 
    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;
 
    if( rowIdx < N && colIdx < N ) {
        C[rowIdx*N + colIdx] = FMA_INTRIN(A[rowIdx*N + colIdx], B[rowIdx*N + colIdx], C[rowIdx*N + colIdx]);
    }
}

#endif // End of GPU_NVIDIA
