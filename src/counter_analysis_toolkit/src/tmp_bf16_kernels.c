#define _GNU_SOURCE
#include <unistd.h>
#include "vec_scalar_verify.h"
#include "cat_arch.h"

#if defined(X86)
#if defined(AVX128_AVAIL)
void tmp_test_bf16_x86_128B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {

    __bf16 one = 1.0f;
    __m128bh vec1;
    __m128bh vec2;
    __m128 result = _mm_set1_ps(0.0);

    int i;
    const int CAP = 8;
    for(i = 0; i < CAP; ++i) {
        vec1[i] = one;
        vec2[i] = one;
    }

    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        fprintf(stderr, "Problem.\n");
        return;
    }

    for(i = 0; i < instr_per_loop; ++i) {
        result = _mm_dpbf16_ps(result, vec1, vec2);
    }

    /*if( result[CAP-1] <= 1.1412 ) {
        fprintf(stderr, "Benchmark artifact. Please ignore.\n");
    }*/
    usleep(1);

    papi_stop_and_print(instr_per_loop, EventSet, fp);

    return;
}

#if defined(AVX256_AVAIL)
void tmp_test_bf16_x86_256B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {

    __bf16 one = 1.0f;
    __m256bh vec1;
    __m256bh vec2;
    __m256 result = _mm256_set1_ps(0.0);

    int i;
    const int CAP = 16;
    for(i = 0; i < 16; ++i) {
        vec1[i] = one;
        vec2[i] = one;
    }

    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        fprintf(stderr, "Problem.\n");
        return;
    }

    for(i = 0; i < instr_per_loop; ++i) {
        result = _mm256_dpbf16_ps(result, vec1, vec2);
    }

    /*if( result[CAP-1] <= 1.1412 ) {
        fprintf(stderr, "Benchmark artifact. Please ignore.\n");
    }*/
    usleep(1);

    papi_stop_and_print(instr_per_loop, EventSet, fp);

    return;
}

#if defined(AVX512_AVAIL)
void tmp_test_bf16_x86_512B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {

    __bf16 one = 1.0f;
    __m512bh vec1;
    __m512bh vec2;
    __m512 result = _mm512_set1_ps(0.0);

    int i;
    const int CAP = 32;
    for(i = 0; i < CAP; ++i) {
        vec1[i] = one;
        vec2[i] = one;
    }

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        fprintf(stderr, "Problem.\n");
        return;
    }

    for(i = 0; i < instr_per_loop; ++i) {
        result = _mm512_dpbf16_ps(result, vec1, vec2);
    }

    /*if( result[CAP-1] <= 1.1412 ) {
        fprintf(stderr, "Benchmark artifact. Please ignore.\n");
    }*/
    usleep(1);

    /* Stop PAPI counters */
    papi_stop_and_print(instr_per_loop, EventSet, fp);

    return;
}
#endif
#endif
#endif
#endif
