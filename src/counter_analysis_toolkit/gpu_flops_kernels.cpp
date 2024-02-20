#include <stdlib.h>
#include <stdio.h>
#include <papi.h>
#include "gpu_flops_kernels.h"

void gpu_matrix_flop(int EventSet, int N, FILE *ofp_papi, int type);

#pragma weak gpu_matrix_flop
void __attribute__((weak)) gpu_matrix_flop(int EventSet, int N, FILE *ofp_papi, int type) {

    fprintf(stderr, "Fell back onto weak symbols for GPU Memory benchmark because it is \
                     not supported on this architecture!\n");
}

/* Wrapper functions of different precisions. */
#if defined(CAT_GPU_PREC_DP)
    extern "C" void gpu_matrix_flop_dp(int EventSet, int N, FILE *ofp_papi, int type) {
        gpu_matrix_flop(EventSet, N, ofp_papi, type);
    }
#elif defined(CAT_GPU_PREC_SP)
    extern "C" void gpu_matrix_flop_sp(int EventSet, int N, FILE *ofp_papi, int type) {
        gpu_matrix_flop(EventSet, N, ofp_papi, type);
    }
#else
    extern "C" void gpu_matrix_flop_hp(int EventSet, int N, FILE *ofp_papi, int type) {
        gpu_matrix_flop(EventSet, N, ofp_papi, type);
    }
#endif

/* GPU kernel prototypes. */
#if defined(GPU_AMD) || defined(GPU_NVIDIA)
__global__ void matrix_add(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N);
__global__ void matrix_sub(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N);
__global__ void matrix_mul(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N);
__global__ void matrix_div(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N);
__global__ void matrix_sqrt(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N);
__global__ void matrix_fma(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N);
//__global__ void matrix_mfma1(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y);
//__global__ void matrix_mfma2(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y);
//__global__ void matrix_mfma3(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y);
//__global__ void matrix_mfma4(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y);
//__host__   void matrix_gemm(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y);
#endif

#if defined(GPU_AMD)
void gpu_matrix_flop(int EventSet, int N, FILE *ofp_papi, int type) {

    int i, j, X, Y, retval;
    CAT_GPU_PREC junk = 0.0;
    long long values = 0;
    size_t probSize, smallProbSize;
    hipError_t status;

    CAT_GPU_PREC  *hostA=NULL,  *hostB=NULL,  *hostC=NULL, *hostD=NULL;
    CAT_GPU_PREC  *copyA=NULL,  *copyB=NULL,  *copyC=NULL, *copyD=NULL;
    CAT_GPU_PREC   *devA=NULL,   *devB=NULL,   *devC=NULL,  *devD=NULL;

    /* Device configuration. */
    if( MFMA1 == type ) {
        X = 16;
        Y = 4;
    } else if( MFMA2 == type ) {
        X = 4;
        Y = 1;
    } else if( MFMA3 == type ) {
        X = 16;
        Y = 1;
    } else if( MFMA4 == type ) {
        X = 4;
        Y = 4;
    } else {
        X = N;
        Y = N;
    }
    dim3 threads_per_block( 1, 1, 1 );
    dim3 threads_per_block_mfma( X, Y, 1 );
    dim3 blocks_in_grid( ceil( float(X) / threads_per_block.x ), ceil( float(Y) / threads_per_block.y ), 1 );
    //dim3 blocks_in_grid_mfma( ceil( float(X) / threads_per_block_mfma.x ), ceil( float(Y) / threads_per_block_mfma.y ), 1 );
    dim3 blocks_in_grid_mfma( ceil( float(X) / threads_per_block.x ), ceil( float(Y) / threads_per_block.y ), 1 );

    smallProbSize = X*Y*sizeof(CAT_GPU_PREC);
    probSize = X*X*sizeof(CAT_GPU_PREC);

    /* Allocate host arrays. */
    status = hipHostMalloc(&hostA, smallProbSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix A.\n");
        return;
    }
    status = hipHostMalloc(&hostB, smallProbSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix B.\n");
        return;
    }
    status = hipHostMalloc(&hostC, probSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix C.\n");
        return;
    }
    status = hipHostMalloc(&hostD, probSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix D.\n");
        return;
    }

    status = hipHostMalloc(&copyA, smallProbSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix A.\n");
        return;
    }
    status = hipHostMalloc(&copyB, smallProbSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix B.\n");
        return;
    }
    status = hipHostMalloc(&copyC, probSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix C.\n");
        return;
    }
    status = hipHostMalloc(&copyD, probSize, 0);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate host matrix D.\n");
        return;
    }

    /* Allocate device arrays. */
    status = hipMalloc(&devA, smallProbSize);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate device matrix A.\n");
        return;
    }
    status = hipMalloc(&devB, smallProbSize);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate device matrix B.\n");
        return;
    }
    status = hipMalloc(&devC, probSize);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate device matrix C.\n");
        return;
    }
    status = hipMalloc(&devD, probSize);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not allocate device matrix D.\n");
        return;
    }

    /* Randomly initialize arrays. */
    srandom(1);
    for( i = 0; i < X; i++ ) {
        for( j = 0; j < Y; j++ ) {
            hostA[i*Y + j] = ((CAT_GPU_PREC)random())/((CAT_GPU_PREC)RAND_MAX) + (CAT_GPU_PREC)1.1;
            copyA[i*Y + j] = hostA[i*Y + j];
        }
        for( j = 0; j < X; j++ ) {
            hostC[i*X + j] = 0.0;
            copyC[i*X + j] = 0.0;
            hostD[i*X + j] = ((CAT_GPU_PREC)random())/((CAT_GPU_PREC)RAND_MAX) + (CAT_GPU_PREC)1.1;
            copyD[i*X + j] = hostD[i*X + j];
        }
    }
    for( i = 0; i < Y; i++ ) {
        for( j = 0; j < X; j++ ) {
            hostB[i*X + j] = ((CAT_GPU_PREC)random())/((CAT_GPU_PREC)RAND_MAX) + (CAT_GPU_PREC)1.1;
            copyB[i*X + j] = hostB[i*X + j];
        }
    }

    /* Copy host data to device. */
    status = hipMemcpy(devA, hostA, smallProbSize, hipMemcpyHostToDevice);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not copy matrix A to device.\n");
        return;
    }
    status = hipMemcpy(devB, hostB, smallProbSize, hipMemcpyHostToDevice);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not copy matrix B to device.\n");
        return;
    }
    status = hipMemcpy(devC, hostC, probSize, hipMemcpyHostToDevice);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not copy matrix C to device.\n");
        return;
    }
    status = hipMemcpy(devD, hostD, probSize, hipMemcpyHostToDevice);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not copy matrix D to device.\n");
        return;
    }

    /* Start PAPI counters. */
    if( (retval = PAPI_start( EventSet )) != PAPI_OK ) return;

    /* Various floating-point operation kernels. */
    //rocblas_handle handle;
    //rocblas_status rstatus;
    //CAT_GPU_PREC alpha = 1.0, beta = 1.0;
    switch(type) {
      case ADD:
          hipLaunchKernelGGL(matrix_add, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
          break;
      case SUB:
          hipLaunchKernelGGL(matrix_sub, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
          break;
      case MUL:
          hipLaunchKernelGGL(matrix_mul, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
          break;
      case DIV:
          hipLaunchKernelGGL(matrix_div, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
          break;
      case SQRT:
          hipLaunchKernelGGL(matrix_sqrt, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
          break;
      case FMA:
          hipLaunchKernelGGL(matrix_fma, blocks_in_grid, threads_per_block, 0, 0, devA, devB, devC, N);
          break;
      /*case MFMA1:
          hipLaunchKernelGGL(matrix_mfma1, blocks_in_grid_mfma, threads_per_block_mfma, 0, 0, devA, devB, devC, devD, X, Y);
          //rstatus = rocblas_create_handle(&handle);
          //#ifndef CAT_GPU_PREC_HP
          //rstatus = RBLAS_GEMM(handle, rocblas_operation_none, rocblas_operation_none, X, X, Y, &alpha, devA, X, devB, Y, &beta, devC, X);
          //#endif
          //rstatus = rocblas_destroy_handle(handle);
          break;
      case MFMA2:
          hipLaunchKernelGGL(matrix_mfma2, blocks_in_grid_mfma, threads_per_block_mfma, 0, 0, devA, devB, devC, devD, X, Y);
          break;
      case MFMA3:
          hipLaunchKernelGGL(matrix_mfma3, blocks_in_grid_mfma, threads_per_block_mfma, 0, 0, devA, devB, devC, devD, X, Y);
          break;
      case MFMA4:
          hipLaunchKernelGGL(matrix_mfma4, blocks_in_grid_mfma, threads_per_block_mfma, 0, 0, devA, devB, devC, devD, X, Y);
          break;*/
      default:
          break;
    }

    /* Error checking. */
    status = hipGetLastError();
    if( hipSuccess != status ) {
        fprintf(stderr, "Error 1.\n");
        return;
    }
    status = hipDeviceSynchronize();
    if( hipSuccess != status ) {
        fprintf(stderr, "Error 2.\n");
        return;
    }

    /* Stop PAPI counters. */
    if( (retval = PAPI_stop(EventSet, &values)) != PAPI_OK ) return;
    fprintf(ofp_papi, "%lld\n", values);

    /* Copy device data to host. */
    status = hipMemcpy(hostC, devC, probSize, hipMemcpyDeviceToHost);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not copy matrix C from device.\n");
        return;
    }

    /* tmp dbg */
    /*if( MFMA1 == type || MFMA2 == type || MFMA3 == type || MFMA4 == type) {
      fprintf(stdout, "\nA:\n");
      for( i = 0; i < X; i++ ) {
        for( j = 0; j < Y; j++ ) {
            fprintf(stdout, "%lf ", (double)hostA[i*Y+j]);
        }
        fprintf(stdout, "\n");
      }
      fprintf(stdout, "\nB:\n");
      for( i = 0; i < Y; i++ ) {
        for( j = 0; j < X; j++ ) {
            fprintf(stdout, "%lf ", (double)hostB[i*X+j]);
        }
        fprintf(stdout, "\n");
      }
      fprintf(stdout, "\nD:\n");
      for( i = 0; i < X; i++ ) {
        for( j = 0; j < X; j++ ) {
            fprintf(stdout, "%lf ", (double)hostD[i*X+j]);
        }
        fprintf(stdout, "\n");
      }
      fprintf(stdout, "\nC [GPU]:\n");
      for( i = 0; i < X; i++ ) {
        for( j = 0; j < X; j++ ) {
            fprintf(stdout, "%lf ", (double)hostC[i*X+j]);
        }
        fprintf(stdout, "\n");
      }

      matrix_gemm(copyA, copyB, copyC, copyD, X, Y);
      fprintf(stdout, "\nC [CPU]:\n");
      for( i = 0; i < X; i++ ) {
        for( j = 0; j < X; j++ ) {
            fprintf(stdout, "%lf ", (double)copyC[i*X+j]);
        }
        fprintf(stdout, "\n");
      }
    }*/

    /* Use the result from the kernels to prevent compiler optimizing it away. */
    junk = (1.23+hostC[X*Y/2])/(1.45+hostC[4*X*Y/5]*hostC[X*Y-1]);
    if( junk > 1.23 && junk < 1.2345 )
        fprintf(stdout, "Benchmark artifact (%f) -- ignore.\n", (float)junk);

    /* Free device memory. */
    status = hipFree(devA);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free device matrix A.\n");
        return;
    }
    status = hipFree(devB);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free device matrix B.\n");
        return;
    }
    status = hipFree(devC);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free device matrix C.\n");
        return;
    }
    status = hipFree(devD);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free device matrix D.\n");
        return;
    }

    /* Free host memory. */
    status = hipFree(hostA);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix A.\n");
        return;
    }
    status = hipFree(hostB);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix B.\n");
        return;
    }
    status = hipFree(hostC);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix C.\n");
        return;
    }
    status = hipFree(hostD);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix D.\n");
        return;
    }

    status = hipFree(copyA);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix A.\n");
        return;
    }
    status = hipFree(copyB);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix B.\n");
        return;
    }
    status = hipFree(copyC);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix C.\n");
        return;
    }
    status = hipFree(copyD);
    if( hipSuccess != status ) {
        fprintf(stderr, "Could not free host matrix D.\n");
        return;
    }
    
    return;
}

__global__ void matrix_add(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        //C[rowIdx*N + colIdx] = A[rowIdx*N + colIdx] + B[rowIdx*N + colIdx];
        //C[rowIdx*N + colIdx] = ADD_INTRIN(A[rowIdx*N + colIdx], B[rowIdx*N + colIdx]);
        __asm__ volatile ( ADD_ASM : "=r" (C[rowIdx*N + colIdx]) : "r" (A[rowIdx*N + colIdx]), "r" (B[rowIdx*N + colIdx]) );
    }
}

__global__ void matrix_sub(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        // this is the subline
        //C[rowIdx*N + colIdx] = A[rowIdx*N + colIdx] - B[rowIdx*N + colIdx];
#if defined(CAT_GPU_PREC_DP)
        __asm__ volatile ( SUB_ASM );
#else
        __asm__ volatile ( SUB_ASM : "=r" (C[rowIdx*N + colIdx]) : "r" (A[rowIdx*N + colIdx]), "r" (B[rowIdx*N + colIdx]) );
#endif
    }
}

__global__ void matrix_mul(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        //C[rowIdx*N + colIdx] = A[rowIdx*N + colIdx] * B[rowIdx*N + colIdx];
        __asm__ volatile ( MUL_ASM : "=r" (C[rowIdx*N + colIdx]) : "r" (A[rowIdx*N + colIdx]), "r" (B[rowIdx*N + colIdx]) );
    }
}

__global__ void matrix_div(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        //C[rowIdx*N + colIdx] = DIV_INTRIN(A[rowIdx*N + colIdx]);
        //__asm__ volatile ( DIV_ASM : "=r" (C[rowIdx*N + colIdx]) : "r" (A[rowIdx*N + colIdx]), "r" (B[rowIdx*N + colIdx]) );
        __asm__ volatile ( DIV_ASM );
    }
}

__global__ void matrix_sqrt(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {

    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;

    if( rowIdx < N && colIdx < N ) {
        //C[rowIdx*N + colIdx] = SQRT_INTRIN(A[rowIdx*N + colIdx]);
        __asm__ volatile ( SQRT_ASM : "=r" (C[rowIdx*N + colIdx]) : "r" (A[rowIdx*N + colIdx]) );
    }
}

__global__ void matrix_fma(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, int N) {
 
    int colIdx = blockDim.x*blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y*blockIdx.y + threadIdx.y;
 
    if( rowIdx < N && colIdx < N ) {
        //C[rowIdx*N + colIdx] = C[rowIdx*N + colIdx] * A[rowIdx*N + colIdx] + B[rowIdx*N + colIdx];
        __asm__ volatile ( FMA_ASM : "=r" (C[rowIdx*N + colIdx]) : "r" (C[rowIdx*N + colIdx]), "r" (A[rowIdx*N + colIdx]), "r" (B[rowIdx*N + colIdx]) );
    }
}

/*__global__ void matrix_mfma1(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y) {

    int i, idx, mk, kn;
    const int LANES = 4;
    using vec4_t = __attribute__( (__vector_size__(LANES * sizeof(CAT_GPU_PREC)) )) CAT_GPU_PREC;
    vec4_t dmn, cmn;

#ifndef CAT_GPU_PREC_HP
    mk = threadIdx.x*Y + threadIdx.y;
    kn = threadIdx.y*X + threadIdx.x;

    for( i = 0; i < LANES; ++i ) {
        idx = threadIdx.x + threadIdx.y*X + i*Y*X;
        dmn[i] = D[idx];
    }

    cmn = MFMA1_INTRIN(A[mk], B[kn], dmn, 0, 0, 0);
 
    for( i = 0; i < LANES; ++i ) {
        idx = threadIdx.x + threadIdx.y*X + i*Y*X;
        C[idx] = cmn[i];
    }
#endif

}

__global__ void matrix_mfma2(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y) {

    int i, idx, mk, kn;
    const int LANES = 4;
    using vec4_t = __attribute__( (__vector_size__(LANES * sizeof(CAT_GPU_PREC)) )) CAT_GPU_PREC;
    vec4_t dmn, cmn;

#if defined(CAT_GPU_PREC_SP)
    mk = threadIdx.x*Y + threadIdx.y;
    kn = threadIdx.y*X + threadIdx.x;

    for( i = 0; i < LANES; ++i ) {
        idx = threadIdx.x + threadIdx.y*X + i*Y*X;
        dmn[i] = D[idx];
    }

    cmn = MFMA2_INTRIN(A[mk], B[kn], dmn, 0, 0, 0);

    for( i = 0; i < LANES; ++i ) {
        idx = threadIdx.x + threadIdx.y*X + i*Y*X;
        C[idx] = cmn[i];
    }
#endif

}

__global__ void matrix_mfma3(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y) {

    int i, idx, mk, kn;
    const int SIMD = 16;
    using vec16_t = __attribute__( (__vector_size__(SIMD * sizeof(CAT_GPU_PREC)) )) CAT_GPU_PREC;
    vec16_t dmn = {0}, cmn;

#if defined(CAT_GPU_PREC_SP)
    mk = threadIdx.x*Y + threadIdx.y;
    kn = threadIdx.y*X + threadIdx.x;

    for( i = 0; i < SIMD; ++i ) {
        idx = threadIdx.x + threadIdx.y*X + i*Y*X;
        dmn[i] = D[idx];
    }

    cmn = MFMA3_INTRIN(A[mk], B[kn], dmn, 0, 0, 0);

    for( i = 0; i < SIMD; ++i ) {
        idx = threadIdx.x + threadIdx.y*X + i*Y*X;
        C[idx] = cmn[i];
    }
#endif

}

__global__ void matrix_mfma4(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y) {

    int i, idx, mk, kn;
    const int LANES = 4;
    using vec4_t = __attribute__( (__vector_size__(LANES * sizeof(CAT_GPU_PREC)) )) CAT_GPU_PREC;
    using vec4f_t = __attribute__( (__vector_size__(LANES * sizeof(float)) )) float;
    vec4_t amn, bmn;
    vec4f_t dmn = {0}, cmn;

#if defined(CAT_GPU_PREC_HP)

    for( i = 0; i < LANES; ++i ) {
        amn[i] = threadIdx.x*Y + i;
        bmn[i] = i*Y + threadIdx.y;
    }

    //for( i = 0; i < LANES; ++i ) {
    //    idx = threadIdx.x + threadIdx.y*X + i*Y*X;
    //    dmn[i] = D[idx];
    //}

    cmn = MFMA4_INTRIN(amn, bmn, dmn, 0, 0, 0);
 
    //for( i = 0; i < LANES; ++i ) {
    //    idx = threadIdx.x + threadIdx.y*X + i*Y*X;
    //    C[idx] = cmn[i];
    //}
    for( i = 0; i < LANES; ++i ) {
        idx = threadIdx.x + threadIdx.y*X + i*Y*X;
        C[idx] = cmn[i];
    }
#endif

}

__host__   void matrix_gemm(CAT_GPU_PREC *A, CAT_GPU_PREC *B, CAT_GPU_PREC *C, CAT_GPU_PREC *D, int X, int Y) {

    int i, j, k;

    for( i = 0; i < X; ++i ) {
        for( j = 0; j < X; ++j ) {
            for( k = 0; k < Y; ++k ) {
                C[i*X + j] += A[i*Y + k] * B[k*X + j];
            }
            C[i*X + j] += D[i*X + j];
        }
    }

}*/

#endif // End of GPU_AMD
