#ifndef _GPU_FLOPS_KERNELS_
#define _GPU_FLOPS_KERNELS_

#include "cat_gpu_arch.h"

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

#endif
