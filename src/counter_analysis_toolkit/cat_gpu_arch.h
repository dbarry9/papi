#define ADD 0
#define SUB 1
#define MUL 2
#define DIV 3
#define SQRT 4
#define FMA 5
#define MFMA1 6
#define MFMA2 7
#define MFMA3 8
#define MFMA4 9

#if defined(GPU_AMD)
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#if defined(CAT_GPU_PREC_DP)
    typedef double CAT_GPU_PREC;
    #define ADD_ASM  "v_add_f64 %1, %2, %0 ;"
    #define SUB_ASM  "v_nop ;"
    #define MUL_ASM  "v_mul_f64 %1, %2, %0 ;"
    #define DIV_ASM  "v_nop ;"
    #define SQRT_ASM "v_sqrt_f64 %1, %0"
    #define FMA_ASM  "v_fma_f64 %1, %2, %0, %3 ;"
    #define DIV_INTRIN(_I_) _I_
    #define SQRT_INTRIN(_I_) __builtin_amdgcn_sqrt(_I_)
    #define MFMA1_INTRIN(_I_, _J_, _K_, _L_, _M_, _N_) __builtin_amdgcn_mfma_f64_16x16x4f64(_I_, _J_, _K_, _L_, _M_, _N_)
    #define RBLAS_GEMM(_I_, _J_, _K_, _L_, _M_, _N_, _O_, _P_, _Q_, _R_, _S_, _T_, _U_, _V_) rocblas_dgemm(_I_, _J_, _K_, _L_, _M_, _N_, _O_, _P_, _Q_, _R_, _S_, _T_, _U_, _V_)
#elif defined(CAT_GPU_PREC_SP)
    typedef float CAT_GPU_PREC;
    #define ADD_ASM  "v_add_f32 %1, %2, %0 ;"
    #define SUB_ASM  "v_sub_f32 %1, %2, %0 ;"
    #define MUL_ASM  "v_mul_f32 %1, %2, %0 ;"
    #define DIV_ASM  "v_nop ;"
    #define SQRT_ASM "v_sqrt_f32 %1, %0"
    #define FMA_ASM  "v_fma_f32 %1, %2, %0, %3 ;"
    #define DIV_INTRIN(_I_) _I_
    #define SQRT_INTRIN(_I_) __builtin_amdgcn_sqrtf(_I_)
    #define MFMA1_INTRIN(_I_, _J_, _K_, _L_, _M_, _N_) __builtin_amdgcn_mfma_f32_16x16x4f32( _I_ , _J_ , _K_ , _L_ , _M_ , _N_ )
    #define MFMA2_INTRIN(_I_, _J_, _K_, _L_, _M_, _N_) __builtin_amdgcn_mfma_f32_4x4x1f32( _I_ , _J_ , _K_ , _L_ , _M_ , _N_ )
    #define MFMA3_INTRIN(_I_, _J_, _K_, _L_, _M_, _N_) __builtin_amdgcn_mfma_f32_16x16x1f32( _I_ , _J_ , _K_ , _L_ , _M_ , _N_ )
    #define RBLAS_GEMM(_I_, _J_, _K_, _L_, _M_, _N_, _O_, _P_, _Q_, _R_, _S_, _T_, _U_, _V_) rocblas_sgemm(_I_, _J_, _K_, _L_, _M_, _N_, _O_, _P_, _Q_, _R_, _S_, _T_, _U_, _V_)
#elif defined(CAT_GPU_PREC_HP)
    typedef _Float16 CAT_GPU_PREC;
    #define ADD_ASM  "v_add_f16 %1, %2, %0 ;"
    #define SUB_ASM  "v_sub_f16 %1, %2, %0 ;"
    #define MUL_ASM  "v_mul_f16 %1, %2, %0 ;"
    #define DIV_ASM  "v_nop ;"
    #define SQRT_ASM "v_sqrt_f16 %1, %0"
    #define FMA_ASM  "v_fma_f16 %1, %2, %0, %3 ;"
    #define ADD_INTRIN(_I_, _J_) __builtin_amdgcn_fadd_f16(_I_, _J_)
    #define DIV_INTRIN(_I_) _I_
    #define SQRT_INTRIN(_I_) __builtin_amdgcn_sqrth(_I_)
    #define MFMA1_INTRIN(_I_, _J_, _K_, _L_, _M_, _N_) __builtin_amdgcn_mfma_f32_16x16x4f16( _I_ , _J_ , _K_ , _L_ , _M_ , _N_ )
    #define MFMA4_INTRIN(_I_, _J_, _K_, _L_, _M_, _N_) __builtin_amdgcn_mfma_f32_4x4x4f16( _I_ , _J_ , _K_ , _L_ , _M_ , _N_ )
    #define RBLAS_GEMM(_I_, _J_, _K_, _L_, _M_, _N_, _O_, _P_, _Q_, _R_, _S_, _T_, _U_, _V_) rocblas_hgemm(_I_, _J_, _K_, _L_, _M_, _N_, _O_, _P_, _Q_, _R_, _S_, _T_, _U_, _V_)
#endif
#endif // End of GPU_AMD

#if defined(GPU_NVIDIA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if defined(CAT_GPU_PREC_DP)
    typedef double CAT_GPU_PREC;
    #define ADD_INTRIN(_I_, _J_) __hadd(_I_, _J_)
    #define SUB_INTRIN(_I_, _J_) __hsub(_I_, _J_)
    #define MUL_INTRIN(_I_, _J_) __hmul(_I_, _J_)
    #define DIV_INTRIN(_I_, _J_) __hdiv(_I_, _J_)
    #define SQRT_INTRIN(_I_) hsqrt(_I_)
    #define FMA_INTRIN(_I_, _J_, _K_) __hfma(_I_, _J_, _K_)
#elif defined(CAT_GPU_PREC_SP)
    typedef float CAT_GPU_PREC;
    #define ADD_INTRIN(_I_, _J_) __fadd_rn(_I_, _J_)
    #define SUB_INTRIN(_I_, _J_) __fsub_rn(_I_, _J_)
    #define MUL_INTRIN(_I_, _J_) __fmul_rn(_I_, _J_)
    #define DIV_INTRIN(_I_, _J_) __fdiv_rn(_I_, _J_)
    #define SQRT_INTRIN(_I_) __fsqrt_rn(_I_)
    #define FMA_INTRIN(_I_, _J_, _K_) __fmaf_rn(_I_, _J_, _K_)
#elif defined(CAT_GPU_PREC_HP)
    typedef half CAT_GPU_PREC;
    #define ADD_INTRIN(_I_, _J_) __dadd_rn(_I_, _J_)
    #define SUB_INTRIN(_I_, _J_) __dsub_rn(_I_, _J_)
    #define MUL_INTRIN(_I_, _J_) __dmul_rn(_I_, _J_)
    #define DIV_INTRIN(_I_, _J_) __ddiv_rn(_I_, _J_)
    #define SQRT_INTRIN(_I_) __dsqrt_rn(_I_)
    #define FMA_INTRIN(_I_, _J_, _K_) __fma_rn(_I_, _J_, _K_)
#endif
#endif // End of GPU_NVIDIA
