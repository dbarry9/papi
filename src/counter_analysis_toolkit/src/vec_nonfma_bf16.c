#include "vec_scalar_verify.h"

#if defined(BF16_AVAIL) //|| (defined(AVX512_BF16_AVAIL) && defined(X86_VEC_WIDTH_512B))
static bf16_half test_bf16_mac_VEC_24( uint64 iterations, int EventSet, FILE *fp );
static bf16_half test_bf16_mac_VEC_48( uint64 iterations, int EventSet, FILE *fp );
static bf16_half test_bf16_mac_VEC_96( uint64 iterations, int EventSet, FILE *fp );
#else
static float test_bf16_mac_VEC_24( uint64 iterations, int EventSet, FILE *fp );
static float test_bf16_mac_VEC_48( uint64 iterations, int EventSet, FILE *fp );
static float test_bf16_mac_VEC_96( uint64 iterations, int EventSet, FILE *fp );
#endif
static void  test_bf16_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_bf16_x86_128B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_bf16_VEC( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_bf16_x86_512B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_bf16_VEC( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_bf16_x86_256B_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_bf16_VEC( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(ARM)
void test_bf16_arm_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_bf16_VEC( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(POWER)
void test_bf16_power_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_bf16_VEC( instr_per_loop, iterations, EventSet, fp );
}
#endif

#if defined(BF16_AVAIL) //|| (defined(AVX512_BF16_AVAIL) && defined(X86_VEC_WIDTH_512B))
/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
static
bf16_half test_bf16_mac_VEC_24( uint64 iterations, int EventSet, FILE *fp ){
    register BF16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PBF16(0.01);
    r1 = SET_VEC_PBF16(0.02);
    r2 = SET_VEC_PBF16(0.03);
    r3 = SET_VEC_PBF16(0.04);
    r4 = SET_VEC_PBF16(0.05);
    r5 = SET_VEC_PBF16(0.06);
    r6 = SET_VEC_PBF16(0.07);
    r7 = SET_VEC_PBF16(0.08);
    r8 = SET_VEC_PBF16(0.09);
    r9 = SET_VEC_PBF16(0.10);
    rA = SET_VEC_PBF16(0.11);
    rB = SET_VEC_PBF16(0.12);
    rC = SET_VEC_PBF16(0.13);
    rD = SET_VEC_PBF16(0.14);
    rE = SET_VEC_PBF16(0.15);
    rF = SET_VEC_PBF16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        fprintf(stderr, "PAPI failed to start.\n");
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < ITER){
            /* The performance critical part */

            r0 = MUL_VEC_PBF16(r0,rC);
            r1 = ADD_VEC_PBF16(r1,rD);
            r2 = MUL_VEC_PBF16(r2,rE);
            r3 = ADD_VEC_PBF16(r3,rF);
            r4 = MUL_VEC_PBF16(r4,rC);
            r5 = ADD_VEC_PBF16(r5,rD);
            r6 = MUL_VEC_PBF16(r6,rE);
            r7 = ADD_VEC_PBF16(r7,rF);
            r8 = MUL_VEC_PBF16(r8,rC);
            r9 = ADD_VEC_PBF16(r9,rD);
            rA = MUL_VEC_PBF16(rA,rE);
            rB = ADD_VEC_PBF16(rB,rF);

            r0 = ADD_VEC_PBF16(r0,rF);
            r1 = MUL_VEC_PBF16(r1,rE);
            r2 = ADD_VEC_PBF16(r2,rD);
            r3 = MUL_VEC_PBF16(r3,rC);
            r4 = ADD_VEC_PBF16(r4,rF);
            r5 = MUL_VEC_PBF16(r5,rE);
            r6 = ADD_VEC_PBF16(r6,rD);
            r7 = MUL_VEC_PBF16(r7,rC);
            r8 = ADD_VEC_PBF16(r8,rF);
            r9 = MUL_VEC_PBF16(r9,rE);
            rA = ADD_VEC_PBF16(rA,rD);
            rB = MUL_VEC_PBF16(rB,rC);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(24, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PBF16(r0,r1);
    r2 = ADD_VEC_PBF16(r2,r3);
    r4 = ADD_VEC_PBF16(r4,r5);
    r6 = ADD_VEC_PBF16(r6,r7);
    r8 = ADD_VEC_PBF16(r8,r9);
    rA = ADD_VEC_PBF16(rA,rB);

    r0 = ADD_VEC_PBF16(r0,r2);
    r4 = ADD_VEC_PBF16(r4,r6);
    r8 = ADD_VEC_PBF16(r8,rA);

    r0 = ADD_VEC_PBF16(r0,r4);
    r0 = ADD_VEC_PBF16(r0,r8);

    bf16_half out = 0;
    BF16_VEC_TYPE temp = r0;
    out += ((bf16_half*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
static
bf16_half test_bf16_mac_VEC_48( uint64 iterations, int EventSet, FILE *fp ){
    register BF16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PBF16(0.01);
    r1 = SET_VEC_PBF16(0.02);
    r2 = SET_VEC_PBF16(0.03);
    r3 = SET_VEC_PBF16(0.04);
    r4 = SET_VEC_PBF16(0.05);
    r5 = SET_VEC_PBF16(0.06);
    r6 = SET_VEC_PBF16(0.07);
    r7 = SET_VEC_PBF16(0.08);
    r8 = SET_VEC_PBF16(0.09);
    r9 = SET_VEC_PBF16(0.10);
    rA = SET_VEC_PBF16(0.11);
    rB = SET_VEC_PBF16(0.12);
    rC = SET_VEC_PBF16(0.13);
    rD = SET_VEC_PBF16(0.14);
    rE = SET_VEC_PBF16(0.15);
    rF = SET_VEC_PBF16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        fprintf(stderr, "PAPI failed to start.\n");
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < ITER){
            /* The performance critical part */

            r0 = MUL_VEC_PBF16(r0,rC);
            r1 = ADD_VEC_PBF16(r1,rD);
            r2 = MUL_VEC_PBF16(r2,rE);
            r3 = ADD_VEC_PBF16(r3,rF);
            r4 = MUL_VEC_PBF16(r4,rC);
            r5 = ADD_VEC_PBF16(r5,rD);
            r6 = MUL_VEC_PBF16(r6,rE);
            r7 = ADD_VEC_PBF16(r7,rF);
            r8 = MUL_VEC_PBF16(r8,rC);
            r9 = ADD_VEC_PBF16(r9,rD);
            rA = MUL_VEC_PBF16(rA,rE);
            rB = ADD_VEC_PBF16(rB,rF);

            r0 = ADD_VEC_PBF16(r0,rF);
            r1 = MUL_VEC_PBF16(r1,rE);
            r2 = ADD_VEC_PBF16(r2,rD);
            r3 = MUL_VEC_PBF16(r3,rC);
            r4 = ADD_VEC_PBF16(r4,rF);
            r5 = MUL_VEC_PBF16(r5,rE);
            r6 = ADD_VEC_PBF16(r6,rD);
            r7 = MUL_VEC_PBF16(r7,rC);
            r8 = ADD_VEC_PBF16(r8,rF);
            r9 = MUL_VEC_PBF16(r9,rE);
            rA = ADD_VEC_PBF16(rA,rD);
            rB = MUL_VEC_PBF16(rB,rC);

            r0 = MUL_VEC_PBF16(r0,rC);
            r1 = ADD_VEC_PBF16(r1,rD);
            r2 = MUL_VEC_PBF16(r2,rE);
            r3 = ADD_VEC_PBF16(r3,rF);
            r4 = MUL_VEC_PBF16(r4,rC);
            r5 = ADD_VEC_PBF16(r5,rD);
            r6 = MUL_VEC_PBF16(r6,rE);
            r7 = ADD_VEC_PBF16(r7,rF);
            r8 = MUL_VEC_PBF16(r8,rC);
            r9 = ADD_VEC_PBF16(r9,rD);
            rA = MUL_VEC_PBF16(rA,rE);
            rB = ADD_VEC_PBF16(rB,rF);

            r0 = ADD_VEC_PBF16(r0,rF);
            r1 = MUL_VEC_PBF16(r1,rE);
            r2 = ADD_VEC_PBF16(r2,rD);
            r3 = MUL_VEC_PBF16(r3,rC);
            r4 = ADD_VEC_PBF16(r4,rF);
            r5 = MUL_VEC_PBF16(r5,rE);
            r6 = ADD_VEC_PBF16(r6,rD);
            r7 = MUL_VEC_PBF16(r7,rC);
            r8 = ADD_VEC_PBF16(r8,rF);
            r9 = MUL_VEC_PBF16(r9,rE);
            rA = ADD_VEC_PBF16(rA,rD);
            rB = MUL_VEC_PBF16(rB,rC);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(48, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PBF16(r0,r1);
    r2 = ADD_VEC_PBF16(r2,r3);
    r4 = ADD_VEC_PBF16(r4,r5);
    r6 = ADD_VEC_PBF16(r6,r7);
    r8 = ADD_VEC_PBF16(r8,r9);
    rA = ADD_VEC_PBF16(rA,rB);

    r0 = ADD_VEC_PBF16(r0,r2);
    r4 = ADD_VEC_PBF16(r4,r6);
    r8 = ADD_VEC_PBF16(r8,rA);

    r0 = ADD_VEC_PBF16(r0,r4);
    r0 = ADD_VEC_PBF16(r0,r8);

    bf16_half out = 0;
    BF16_VEC_TYPE temp = r0;
    out += ((bf16_half*)&temp)[0];

    return out;
}

/************************************/
/* Loop unrolling:  96 instructions */
/************************************/
static
bf16_half test_bf16_mac_VEC_96( uint64 iterations, int EventSet, FILE *fp ){
    register BF16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PBF16(0.01);
    r1 = SET_VEC_PBF16(0.02);
    r2 = SET_VEC_PBF16(0.03);
    r3 = SET_VEC_PBF16(0.04);
    r4 = SET_VEC_PBF16(0.05);
    r5 = SET_VEC_PBF16(0.06);
    r6 = SET_VEC_PBF16(0.07);
    r7 = SET_VEC_PBF16(0.08);
    r8 = SET_VEC_PBF16(0.09);
    r9 = SET_VEC_PBF16(0.10);
    rA = SET_VEC_PBF16(0.11);
    rB = SET_VEC_PBF16(0.12);
    rC = SET_VEC_PBF16(0.13);
    rD = SET_VEC_PBF16(0.14);
    rE = SET_VEC_PBF16(0.15);
    rF = SET_VEC_PBF16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
        fprintf(stderr, "PAPI failed to start.\n");
        return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < ITER){
            /* The performance critical part */

            r0 = MUL_VEC_PBF16(r0,rC);
            r1 = ADD_VEC_PBF16(r1,rD);
            r2 = MUL_VEC_PBF16(r2,rE);
            r3 = ADD_VEC_PBF16(r3,rF);
            r4 = MUL_VEC_PBF16(r4,rC);
            r5 = ADD_VEC_PBF16(r5,rD);
            r6 = MUL_VEC_PBF16(r6,rE);
            r7 = ADD_VEC_PBF16(r7,rF);
            r8 = MUL_VEC_PBF16(r8,rC);
            r9 = ADD_VEC_PBF16(r9,rD);
            rA = MUL_VEC_PBF16(rA,rE);
            rB = ADD_VEC_PBF16(rB,rF);

            r0 = ADD_VEC_PBF16(r0,rF);
            r1 = MUL_VEC_PBF16(r1,rE);
            r2 = ADD_VEC_PBF16(r2,rD);
            r3 = MUL_VEC_PBF16(r3,rC);
            r4 = ADD_VEC_PBF16(r4,rF);
            r5 = MUL_VEC_PBF16(r5,rE);
            r6 = ADD_VEC_PBF16(r6,rD);
            r7 = MUL_VEC_PBF16(r7,rC);
            r8 = ADD_VEC_PBF16(r8,rF);
            r9 = MUL_VEC_PBF16(r9,rE);
            rA = ADD_VEC_PBF16(rA,rD);
            rB = MUL_VEC_PBF16(rB,rC);

            r0 = MUL_VEC_PBF16(r0,rC);
            r1 = ADD_VEC_PBF16(r1,rD);
            r2 = MUL_VEC_PBF16(r2,rE);
            r3 = ADD_VEC_PBF16(r3,rF);
            r4 = MUL_VEC_PBF16(r4,rC);
            r5 = ADD_VEC_PBF16(r5,rD);
            r6 = MUL_VEC_PBF16(r6,rE);
            r7 = ADD_VEC_PBF16(r7,rF);
            r8 = MUL_VEC_PBF16(r8,rC);
            r9 = ADD_VEC_PBF16(r9,rD);
            rA = MUL_VEC_PBF16(rA,rE);
            rB = ADD_VEC_PBF16(rB,rF);

            r0 = ADD_VEC_PBF16(r0,rF);
            r1 = MUL_VEC_PBF16(r1,rE);
            r2 = ADD_VEC_PBF16(r2,rD);
            r3 = MUL_VEC_PBF16(r3,rC);
            r4 = ADD_VEC_PBF16(r4,rF);
            r5 = MUL_VEC_PBF16(r5,rE);
            r6 = ADD_VEC_PBF16(r6,rD);
            r7 = MUL_VEC_PBF16(r7,rC);
            r8 = ADD_VEC_PBF16(r8,rF);
            r9 = MUL_VEC_PBF16(r9,rE);
            rA = ADD_VEC_PBF16(rA,rD);
            rB = MUL_VEC_PBF16(rB,rC);

            r0 = MUL_VEC_PBF16(r0,rC);
            r1 = ADD_VEC_PBF16(r1,rD);
            r2 = MUL_VEC_PBF16(r2,rE);
            r3 = ADD_VEC_PBF16(r3,rF);
            r4 = MUL_VEC_PBF16(r4,rC);
            r5 = ADD_VEC_PBF16(r5,rD);
            r6 = MUL_VEC_PBF16(r6,rE);
            r7 = ADD_VEC_PBF16(r7,rF);
            r8 = MUL_VEC_PBF16(r8,rC);
            r9 = ADD_VEC_PBF16(r9,rD);
            rA = MUL_VEC_PBF16(rA,rE);
            rB = ADD_VEC_PBF16(rB,rF);

            r0 = ADD_VEC_PBF16(r0,rF);
            r1 = MUL_VEC_PBF16(r1,rE);
            r2 = ADD_VEC_PBF16(r2,rD);
            r3 = MUL_VEC_PBF16(r3,rC);
            r4 = ADD_VEC_PBF16(r4,rF);
            r5 = MUL_VEC_PBF16(r5,rE);
            r6 = ADD_VEC_PBF16(r6,rD);
            r7 = MUL_VEC_PBF16(r7,rC);
            r8 = ADD_VEC_PBF16(r8,rF);
            r9 = MUL_VEC_PBF16(r9,rE);
            rA = ADD_VEC_PBF16(rA,rD);
            rB = MUL_VEC_PBF16(rB,rC);

            r0 = MUL_VEC_PBF16(r0,rC);
            r1 = ADD_VEC_PBF16(r1,rD);
            r2 = MUL_VEC_PBF16(r2,rE);
            r3 = ADD_VEC_PBF16(r3,rF);
            r4 = MUL_VEC_PBF16(r4,rC);
            r5 = ADD_VEC_PBF16(r5,rD);
            r6 = MUL_VEC_PBF16(r6,rE);
            r7 = ADD_VEC_PBF16(r7,rF);
            r8 = MUL_VEC_PBF16(r8,rC);
            r9 = ADD_VEC_PBF16(r9,rD);
            rA = MUL_VEC_PBF16(rA,rE);
            rB = ADD_VEC_PBF16(rB,rF);

            r0 = ADD_VEC_PBF16(r0,rF);
            r1 = MUL_VEC_PBF16(r1,rE);
            r2 = ADD_VEC_PBF16(r2,rD);
            r3 = MUL_VEC_PBF16(r3,rC);
            r4 = ADD_VEC_PBF16(r4,rF);
            r5 = MUL_VEC_PBF16(r5,rE);
            r6 = ADD_VEC_PBF16(r6,rD);
            r7 = MUL_VEC_PBF16(r7,rC);
            r8 = ADD_VEC_PBF16(r8,rF);
            r9 = MUL_VEC_PBF16(r9,rE);
            rA = ADD_VEC_PBF16(rA,rD);
            rB = MUL_VEC_PBF16(rB,rC);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(96, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PBF16(r0,r1);
    r2 = ADD_VEC_PBF16(r2,r3);
    r4 = ADD_VEC_PBF16(r4,r5);
    r6 = ADD_VEC_PBF16(r6,r7);
    r8 = ADD_VEC_PBF16(r8,r9);
    rA = ADD_VEC_PBF16(rA,rB);

    r0 = ADD_VEC_PBF16(r0,r2);
    r4 = ADD_VEC_PBF16(r4,r6);
    r8 = ADD_VEC_PBF16(r8,rA);

    r0 = ADD_VEC_PBF16(r0,r4);
    r0 = ADD_VEC_PBF16(r0,r8);

    bf16_half out = 0;
    BF16_VEC_TYPE temp = r0;
    out += ((bf16_half*)&temp)[0];

    return out;
}

static
void test_bf16_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    bf16_half sum = 0.0;
    bf16_half scalar_sum = 0.0;

    if ( instr_per_loop == 24 ) {
        sum += test_bf16_mac_VEC_24( iterations, EventSet, fp );
        scalar_sum += test_bf16_scalar_VEC_24( iterations, EventSet, NULL );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_bf16_mac_VEC_48( iterations, EventSet, fp );
        scalar_sum += test_bf16_scalar_VEC_48( iterations, EventSet, NULL );
    }
    else if ( instr_per_loop == 96 ) {
        sum += test_bf16_mac_VEC_96( iterations, EventSet, fp );
        scalar_sum += test_bf16_scalar_VEC_96( iterations, EventSet, NULL );
    }

    if( sum != scalar_sum ) {
        fprintf(stderr, "Inconsistent FLOP results detected!\n");
    }
}

#else
static
float test_bf16_mac_VEC_24( uint64 iterations, int EventSet, FILE *fp ){

    (void)iterations;
    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(24, fp);
    }

    return 0.0;
}

static
float test_bf16_mac_VEC_48( uint64 iterations, int EventSet, FILE *fp ){

    (void)iterations;
    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(48, fp);
    }

    return 0.0;
}

static
float test_bf16_mac_VEC_96( uint64 iterations, int EventSet, FILE *fp ){

    (void)iterations;
    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(96, fp);
    }

    return 0.0;
}

static
void test_bf16_VEC( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 24 ) {
        sum += test_bf16_mac_VEC_24( iterations, EventSet, fp );
        scalar_sum += test_bf16_scalar_VEC_24( iterations, EventSet, NULL );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_bf16_mac_VEC_48( iterations, EventSet, fp );
        scalar_sum += test_bf16_scalar_VEC_48( iterations, EventSet, NULL );
    }
    else if ( instr_per_loop == 96 ) {
        sum += test_bf16_mac_VEC_96( iterations, EventSet, fp );
        scalar_sum += test_bf16_scalar_VEC_96( iterations, EventSet, NULL );
    }

    if( sum != scalar_sum ) {
        fprintf(stderr, "Inconsistent FLOP results detected!\n");
    }
}
#endif
