#include <inttypes.h>

typedef unsigned long long uint64;

#if defined(X86)
void test_hp_x86_128B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_x86_128B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_x86_128B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

void test_hp_x86_256B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_x86_256B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_x86_256B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

void test_hp_x86_512B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_x86_512B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_x86_512B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

void test_hp_x86_128B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_x86_128B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_x86_128B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

void test_hp_x86_256B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_x86_256B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_x86_256B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

void test_hp_x86_512B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_sp_x86_512B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void test_dp_x86_512B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

#include <immintrin.h>

//typedef _Float16 BF16_SCALAR_TYPE;
//typedef _Float16 FP16_SCALAR_TYPE;
typedef _Float16 TEMP_TYPE; // Need to rename this
typedef __m128bh BF16_SCALAR_TYPE;
typedef __m128h  FP16_SCALAR_TYPE;
typedef __m128   SP_SCALAR_TYPE;
typedef __m128d  DP_SCALAR_TYPE;

#define SET_VEC_SBF16(_I_)         (_mm_set_sbh( _I_ ));
#define ADD_VEC_SBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm_add_sbh( _I_ , _J_ )[0]);
#define MUL_VEC_SBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm_mul_sbh( _I_ , _J_ )[0]);
#define FMA_VEC_SBF16(_I_,_J_,_K_) (BF16_SCALAR_TYPE*)&(_mm_fmadd_sbh( _I_ , _J_ , _K_ )[0]);
//#define DIV_VEC_SBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm_div_sbh( _I_ , _J_ , _K_ )[0])

#define SET_VEC_SFP16(_I_)         (_mm_set_sh( (_Float16)_I_ ));
#define ADD_VEC_SFP16(_I_,_J_)     (_mm_add_sh( _I_ , _J_ ));
#define MUL_VEC_SFP16(_I_,_J_)     (_mm_mul_sh( _I_ , _J_ ));
#define FMA_VEC_SFP16(_I_,_J_,_K_) (_mm_fmadd_sh( _I_ , _J_ , _K_ ));
//#define DIV_VEC_SFP16(_I_,_J_)     (_mm_div_sh( _I_ , _J_ ))

#define SET_VEC_SS(_I_)         _mm_set_ss( _I_ );
#define ADD_VEC_SS(_I_,_J_)     _mm_add_ss( _I_ , _J_ );
#define MUL_VEC_SS(_I_,_J_)     _mm_mul_ss( _I_ , _J_ );
#define FMA_VEC_SS(_I_,_J_,_K_) _mm_fmadd_ss( _I_ , _J_ , _K_ );

#define SET_VEC_SD(_I_)         _mm_set_sd( _I_ );
#define ADD_VEC_SD(_I_,_J_)     _mm_add_sd( _I_ , _J_ );
#define MUL_VEC_SD(_I_,_J_)     _mm_mul_sd( _I_ , _J_ );
#define FMA_VEC_SD(_I_,_J_,_K_) _mm_fmadd_sd( _I_ , _J_ , _K_ );

#if defined(X86_VEC_WIDTH_128B)
typedef __m128bh BF16_VEC_TYPE;
typedef __m128h  FP16_VEC_TYPE;
typedef __m128     SP_VEC_TYPE;
typedef __m128d    DP_VEC_TYPE;

#define SET_VEC_PBF16(_I_)         (BF16_SCALAR_TYPE*)&(_mm_set1_pbh( _I_ )[0]);
#define ADD_VEC_PBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm_add_pbh( _I_ , _J_ )[0]);
#define MUL_VEC_PBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm_mul_pbh( _I_ , _J_ )[0]);
#define FMA_VEC_PBF16(_I_,_J_,_K_) (BF16_SCALAR_TYPE*)&(_mm_fmadd_pbh( _I_ , _J_ , _K_ )[0]);

#define SET_VEC_PFP16(_I_)         (_mm_set1_ph( (_Float16)_I_ ));
#define ADD_VEC_PFP16(_I_,_J_)     (_mm_add_ph( _I_ , _J_ ));
#define MUL_VEC_PFP16(_I_,_J_)     (_mm_mul_ph( _I_ , _J_ ));
#define FMA_VEC_PFP16(_I_,_J_,_K_) (_mm_fmadd_ph( _I_ , _J_ , _K_ ));

#define SET_VEC_PS(_I_)         _mm_set1_ps( _I_ );
#define ADD_VEC_PS(_I_,_J_)     _mm_add_ps( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_)     _mm_mul_ps( _I_ , _J_ );
#define FMA_VEC_PS(_I_,_J_,_K_) _mm_fmadd_ps( _I_ , _J_ , _K_ );

#define SET_VEC_PD(_I_)         _mm_set1_pd( _I_ );
#define ADD_VEC_PD(_I_,_J_)     _mm_add_pd( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)     _mm_mul_pd( _I_ , _J_ );
#define FMA_VEC_PD(_I_,_J_,_K_) _mm_fmadd_pd( _I_ , _J_ , _K_ );

#elif defined(X86_VEC_WIDTH_512B)
typedef __m512bh BF16_VEC_TYPE;
typedef __m512h  FP16_VEC_TYPE;
typedef __m512     SP_VEC_TYPE;
typedef __m512d    DP_VEC_TYPE;

#define SET_VEC_PBF16(_I_)         (BF16_SCALAR_TYPE*)&(_mm512_set1_pbh( _I_ )[0]);
#define ADD_VEC_PBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm512_add_pbh( _I_ , _J_ )[0]);
#define MUL_VEC_PBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm512_mul_pbh( _I_ , _J_ )[0]);
#define FMA_VEC_PBF16(_I_,_J_,_K_) (BF16_SCALAR_TYPE*)&(_mm512_fmadd_pbh( _I_ , _J_ , _K_ )[0]);

#define SET_VEC_PFP16(_I_)         (_mm512_set1_ph( (_Float16)_I_ ));
#define ADD_VEC_PFP16(_I_,_J_)     (_mm512_add_ph( _I_ , _J_ ));
#define MUL_VEC_PFP16(_I_,_J_)     (_mm512_mul_ph( _I_ , _J_ ));
#define FMA_VEC_PFP16(_I_,_J_,_K_) (_mm512_fmadd_ph( _I_ , _J_ , _K_ ));

#define SET_VEC_PS(_I_)         _mm512_set1_ps( _I_ );
#define ADD_VEC_PS(_I_,_J_)     _mm512_add_ps( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_)     _mm512_mul_ps( _I_ , _J_ );
#define FMA_VEC_PS(_I_,_J_,_K_) _mm512_fmadd_ps( _I_ , _J_ , _K_ );

#define SET_VEC_PD(_I_)         _mm512_set1_pd( _I_ );
#define ADD_VEC_PD(_I_,_J_)     _mm512_add_pd( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)     _mm512_mul_pd( _I_ , _J_ );
#define FMA_VEC_PD(_I_,_J_,_K_) _mm512_fmadd_pd( _I_ , _J_ , _K_ );

#else
typedef __m256bh BF16_VEC_TYPE;
typedef __m256h  FP16_VEC_TYPE;
typedef __m256     SP_VEC_TYPE;
typedef __m256d    DP_VEC_TYPE;

#define SET_VEC_PBF16(_I_)         (BF16_SCALAR_TYPE*)&(_mm256_set1_pbh( _I_ )[0]);
#define ADD_VEC_PBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm256_add_pbh( _I_ , _J_ )[0]);
#define MUL_VEC_PBF16(_I_,_J_)     (BF16_SCALAR_TYPE*)&(_mm256_mul_pbh( _I_ , _J_ )[0]);
#define FMA_VEC_PBF16(_I_,_J_,_K_) (BF16_SCALAR_TYPE*)&(_mm256_fmadd_pbh( _I_ , _J_ , _K_ )[0]);

#define SET_VEC_PFP16(_I_)         (_mm256_set1_ph( (_Float16)_I_ ));
#define ADD_VEC_PFP16(_I_,_J_)     (_mm256_add_ph( _I_ , _J_ ));
#define MUL_VEC_PFP16(_I_,_J_)     (_mm256_mul_ph( _I_ , _J_ ));
#define FMA_VEC_PFP16(_I_,_J_,_K_) (_mm256_fmadd_ph( _I_ , _J_ , _K_ ));

#define SET_VEC_PS(_I_)         _mm256_set1_ps( _I_ );
#define ADD_VEC_PS(_I_,_J_)     _mm256_add_ps( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_)     _mm256_mul_ps( _I_ , _J_ );
#define FMA_VEC_PS(_I_,_J_,_K_) _mm256_fmadd_ps( _I_ , _J_ , _K_ );

#define SET_VEC_PD(_I_)         _mm256_set1_pd( _I_ );
#define ADD_VEC_PD(_I_,_J_)     _mm256_add_pd( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)     _mm256_mul_pd( _I_ , _J_ );
#define FMA_VEC_PD(_I_,_J_,_K_) _mm256_fmadd_pd( _I_ , _J_ , _K_ );
#endif

#elif defined(ARM)
void  test_hp_arm_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_sp_arm_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_dp_arm_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_hp_arm_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_sp_arm_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_dp_arm_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

#include <arm_neon.h>

typedef __bf16 BF16_SCALAR_TYPE;
typedef __fp16 FP16_SCALAR_TYPE;
typedef float  SP_SCALAR_TYPE;
typedef double DP_SCALAR_TYPE;
typedef bfloat16x8_t BF16_VEC_TYPE;
typedef float16x8_t  FP16_VEC_TYPE;
typedef float32x4_t    SP_VEC_TYPE;
typedef float64x2_t    DP_VEC_TYPE;

#define SET_VEC_PBF16(_I_) (BF16_VEC_TYPE)vdupq_n_bf16( _I_ );
#define SET_VEC_PFP16(_I_) (FP16_VEC_TYPE)vdupq_n_f16( _I_ );
#define SET_VEC_PS(_I_)      (SP_VEC_TYPE)vdupq_n_f32( _I_ );
#define SET_VEC_PD(_I_)      (DP_VEC_TYPE)vdupq_n_f64( _I_ );

#define ADD_VEC_PBF16(_I_,_J_) (BF16_VEC_TYPE)vaddq_bf16( _I_ , _J_ );
#define ADD_VEC_PFP16(_I_,_J_) (FP16_VEC_TYPE)vaddq_f16( _I_ , _J_ );
#define ADD_VEC_PS(_I_,_J_)      (SP_VEC_TYPE)vaddq_f32( _I_ , _J_ );
#define ADD_VEC_PD(_I_,_J_)      (DP_VEC_TYPE)vaddq_f64( _I_ , _J_ );

#define MUL_VEC_PBF16(_I_,_J_) (BF16_VEC_TYPE)vmulq_bf16( _I_ , _J_ );
#define MUL_VEC_PFP16(_I_,_J_) (FP16_VEC_TYPE)vmulq_f16( _I_ , _J_ );
#define MUL_VEC_PS(_I_,_J_)      (SP_VEC_TYPE)vmulq_f32( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_)      (DP_VEC_TYPE)vmulq_f64( _I_ , _J_ );

#define FMA_VEC_PBF16(_I_,_J_,_K_) (BF16_VEC_TYPE)vfmaq_bf16( _K_ , _J_ , _I_ );
#define FMA_VEC_PFP16(_I_,_J_,_K_) (FP16_VEC_TYPE)vfmaq_f16( _K_ , _J_ , _I_ );
#define FMA_VEC_PS(_I_,_J_,_K_)      (SP_VEC_TYPE)vfmaq_f32( _K_ , _J_ , _I_ );
#define FMA_VEC_PD(_I_,_J_,_K_)      (DP_VEC_TYPE)vfmaq_f64( _K_ , _J_ , _I_ );

/* There is no scalar FMA intrinsic available on this architecture. */
#define SET_VEC_SBF16(_I_)         _I_ ;
#define ADD_VEC_SBF16(_I_,_J_)     _I_ + _J_;
#define MUL_VEC_SBF16(_I_,_J_)     _I_ * _J_ ;
//#define SQRT_VEC_SBF16(_I_)        vsqrth_f16( _I_ );
#define FMA_VEC_SBF16(_I_,_J_,_K_) _I_ * _J_ + _K_;
//#define DIV_VEC_SBF16(_I_,_J_)     _I_ / _J_

#define SET_VEC_SFP16(_I_)         _I_ ;
#define ADD_VEC_SFP16(_I_,_J_)     vaddh_f16( _I_ , _J_ );
#define MUL_VEC_SFP16(_I_,_J_)     vmulh_f16( _I_ , _J_ );
#define SQRT_VEC_SFP16(_I_)        vsqrth_f16( _I_ );
#define FMA_VEC_SFP16(_I_,_J_,_K_) _I_ * _J_ + _K_;
//#define DIV_VEC_SFP16(_I_,_J_)     vdivh_f16( _I_ , _J_ )

#define SET_VEC_SS(_I_)         _I_ ;
#define ADD_VEC_SS(_I_,_J_)     _I_ + _J_ ;
#define MUL_VEC_SS(_I_,_J_)     _I_ * _J_ ;
#define FMA_VEC_SS(_I_,_J_,_K_) _I_ * _J_ + _K_;

#define SET_VEC_SD(_I_)         _I_ ;
#define ADD_VEC_SD(_I_,_J_)     _I_ + _J_ ;
#define MUL_VEC_SD(_I_,_J_)     _I_ * _J_ ;
#define FMA_VEC_SD(_I_,_J_,_K_) _I_ * _J_ + _K_;

#elif defined(POWER)
void  test_hp_power_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_sp_power_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_dp_power_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_hp_power_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_sp_power_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );
void  test_dp_power_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

#include <altivec.h>

typedef float  SP_SCALAR_TYPE;
typedef double DP_SCALAR_TYPE;
typedef __vector float  SP_VEC_TYPE;
typedef __vector double DP_VEC_TYPE;

#define SET_VEC_PS(_I_) (SP_VEC_TYPE){ _I_ , _I_ , _I_ , _I_ };
#define SET_VEC_PD(_I_) (DP_VEC_TYPE){ _I_ , _I_ };

#define ADD_VEC_PS(_I_,_J_) (SP_VEC_TYPE)vec_add( _I_ , _J_ );
#define ADD_VEC_PD(_I_,_J_) (DP_VEC_TYPE)vec_add( _I_ , _J_ );

#define MUL_VEC_PS(_I_,_J_) (SP_VEC_TYPE)vec_mul( _I_ , _J_ );
#define MUL_VEC_PD(_I_,_J_) (DP_VEC_TYPE)vec_mul( _I_ , _J_ );

#define FMA_VEC_PS(_I_,_J_,_K_) (SP_VEC_TYPE)vec_madd( _I_ , _J_ , _K_ );
#define FMA_VEC_PD(_I_,_J_,_K_) (DP_VEC_TYPE)vec_madd( _I_ , _J_ , _K_ );

/* There is no scalar FMA intrinsic available on this architecture. */
#define SET_VEC_SS(_I_)         _I_ ;
#define ADD_VEC_SS(_I_,_J_)     _I_ + _J_ ;
#define MUL_VEC_SS(_I_,_J_)     _I_ * _J_ ;
#define FMA_VEC_SS(_I_,_J_,_K_) _I_ * _J_ + _K_;

#define SET_VEC_SD(_I_)         _I_ ;
#define ADD_VEC_SD(_I_,_J_)     _I_ + _J_ ;
#define MUL_VEC_SD(_I_,_J_)     _I_ * _J_ ;
#define FMA_VEC_SD(_I_,_J_,_K_) _I_ * _J_ + _K_;

#endif
