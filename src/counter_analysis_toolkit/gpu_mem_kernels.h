#ifndef _GPU_MEM_KERNELS_
#define _GPU_MEM_KERNELS_

#if defined(GPU_AMD)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#endif // End of GPU_AMD

#if defined(GPU_NVIDIA)
#include <cuda_runtime.h>
#endif // End of GPU_NVIDIA

#endif
