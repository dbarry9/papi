#include "vec_scalar_verify.h"

#if defined(ARM) || defined(X86)
static FP16_SCALAR_TYPE  test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
static FP16_SCALAR_TYPE  test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
static FP16_SCALAR_TYPE  test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );
#else
static float test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp );
static float test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp );
static float test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp );
#endif
static void  test_hp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp );

/* Wrapper functions of different vector widths. */
#if defined(X86_VEC_WIDTH_128B)
void test_hp_x86_128B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_hp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_512B)
void test_hp_x86_512B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_hp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(X86_VEC_WIDTH_256B)
void test_hp_x86_256B_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_hp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(ARM)
void test_hp_arm_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_hp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#elif defined(POWER)
void test_hp_power_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp ) {
    return test_hp_VEC_FMA( instr_per_loop, iterations, EventSet, fp );
}
#endif

#if defined(ARM) || defined(X86)
/************************************/
/* Loop unrolling:  12 instructions */
/************************************/
static
FP16_SCALAR_TYPE test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp ){
    register FP16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PFP16(0.01);
    r1 = SET_VEC_PFP16(0.02);
    r2 = SET_VEC_PFP16(0.03);
    r3 = SET_VEC_PFP16(0.04);
    r4 = SET_VEC_PFP16(0.05);
    r5 = SET_VEC_PFP16(0.06);
    r6 = SET_VEC_PFP16(0.07);
    r7 = SET_VEC_PFP16(0.08);
    r8 = SET_VEC_PFP16(0.09);
    r9 = SET_VEC_PFP16(0.10);
    rA = SET_VEC_PFP16(0.11);
    rB = SET_VEC_PFP16(0.12);
    rC = SET_VEC_PFP16(0.13);
    rD = SET_VEC_PFP16(0.14);
    rE = SET_VEC_PFP16(0.15);
    rF = SET_VEC_PFP16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
	fprintf(stderr, "PAPI failed to start, but skipping return for dev purposes.\n");
        //return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < ITER){
        /* The performance critical part */

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(12, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PFP16(r0,r1);
    r2 = ADD_VEC_PFP16(r2,r3);
    r4 = ADD_VEC_PFP16(r4,r5);

    r0 = ADD_VEC_PFP16(r0,r6);
    r2 = ADD_VEC_PFP16(r2,r4);

    r0 = ADD_VEC_PFP16(r0,r2);

    FP16_SCALAR_TYPE out = SET_VEC_SFP16(0);
    FP16_VEC_TYPE temp = r0;
    out = ADD_VEC_SFP16(out,((FP16_SCALAR_TYPE*)&temp)[0]);

    return out;
}

/************************************/
/* Loop unrolling:  24 instructions */
/************************************/
static
FP16_SCALAR_TYPE test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp ){
    register FP16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PFP16(0.01);
    r1 = SET_VEC_PFP16(0.02);
    r2 = SET_VEC_PFP16(0.03);
    r3 = SET_VEC_PFP16(0.04);
    r4 = SET_VEC_PFP16(0.05);
    r5 = SET_VEC_PFP16(0.06);
    r6 = SET_VEC_PFP16(0.07);
    r7 = SET_VEC_PFP16(0.08);
    r8 = SET_VEC_PFP16(0.09);
    r9 = SET_VEC_PFP16(0.10);
    rA = SET_VEC_PFP16(0.11);
    rB = SET_VEC_PFP16(0.12);
    rC = SET_VEC_PFP16(0.13);
    rD = SET_VEC_PFP16(0.14);
    rE = SET_VEC_PFP16(0.15);
    rF = SET_VEC_PFP16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
	fprintf(stderr, "PAPI failed to start, but skipping return for dev purposes.\n");
        //return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < ITER){
            /* The performance critical part */

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(24, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PFP16(r0,r1);
    r2 = ADD_VEC_PFP16(r2,r3);
    r4 = ADD_VEC_PFP16(r4,r5);

    r0 = ADD_VEC_PFP16(r0,r6);
    r2 = ADD_VEC_PFP16(r2,r4);

    r0 = ADD_VEC_PFP16(r0,r2);

    FP16_SCALAR_TYPE out = SET_VEC_SFP16(0);
    FP16_VEC_TYPE temp = r0;
    out = ADD_VEC_SFP16(out,((FP16_SCALAR_TYPE*)&temp)[0]);

    return out;
}

/************************************/
/* Loop unrolling:  48 instructions */
/************************************/
static
FP16_SCALAR_TYPE test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp ){
    register FP16_VEC_TYPE r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;

    /* Generate starting data */
    r0 = SET_VEC_PFP16(0.01);
    r1 = SET_VEC_PFP16(0.02);
    r2 = SET_VEC_PFP16(0.03);
    r3 = SET_VEC_PFP16(0.04);
    r4 = SET_VEC_PFP16(0.05);
    r5 = SET_VEC_PFP16(0.06);
    r6 = SET_VEC_PFP16(0.07);
    r7 = SET_VEC_PFP16(0.08);
    r8 = SET_VEC_PFP16(0.09);
    r9 = SET_VEC_PFP16(0.10);
    rA = SET_VEC_PFP16(0.11);
    rB = SET_VEC_PFP16(0.12);
    rC = SET_VEC_PFP16(0.13);
    rD = SET_VEC_PFP16(0.14);
    rE = SET_VEC_PFP16(0.15);
    rF = SET_VEC_PFP16(0.16);

    /* Start PAPI counters */
    if ( PAPI_start( EventSet ) != PAPI_OK ) {
	fprintf(stderr, "PAPI failed to start, but skipping return for dev purposes.\n");
        //return -1;
    }

    uint64 c = 0;
    while (c < iterations){
        size_t i = 0;
        while (i < ITER){
            /* The performance critical part */

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            r0 = FMA_VEC_PFP16(r0,r7,r9);
            r1 = FMA_VEC_PFP16(r1,r8,rA);
            r2 = FMA_VEC_PFP16(r2,r9,rB);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,rB,rD);
            r5 = FMA_VEC_PFP16(r5,rC,rE);

            r0 = FMA_VEC_PFP16(r0,rD,rF);
            r1 = FMA_VEC_PFP16(r1,rC,rE);
            r2 = FMA_VEC_PFP16(r2,rB,rD);
            r3 = FMA_VEC_PFP16(r3,rA,rC);
            r4 = FMA_VEC_PFP16(r4,r9,rB);
            r5 = FMA_VEC_PFP16(r5,r8,rA);

            i++;
        }
        c++;
    }

    /* Stop PAPI counters */
    papi_stop_and_print(48, EventSet, fp);

    /* Use data so that compiler does not eliminate it when using -O2 */
    r0 = ADD_VEC_PFP16(r0,r1);
    r2 = ADD_VEC_PFP16(r2,r3);
    r4 = ADD_VEC_PFP16(r4,r5);

    r0 = ADD_VEC_PFP16(r0,r6);
    r2 = ADD_VEC_PFP16(r2,r4);

    r0 = ADD_VEC_PFP16(r0,r2);

    FP16_SCALAR_TYPE out = SET_VEC_SFP16(0);
    FP16_VEC_TYPE temp = r0;
    out = ADD_VEC_SFP16(out,((FP16_SCALAR_TYPE*)&temp)[0]); // this is redundant -- needlessly complex

    return out;
}

static
void test_hp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    FP16_SCALAR_TYPE sum = SET_VEC_SFP16(0.0);
    FP16_SCALAR_TYPE scalar_sum = SET_VEC_SFP16(0.0);

    if ( instr_per_loop == 12 ) {
        sum = ADD_VEC_SFP16(sum,test_hp_mac_VEC_FMA_12( iterations, EventSet, fp ));
        scalar_sum = ADD_VEC_SFP16(scalar_sum,test_hp_scalar_VEC_FMA_12( iterations, EventSet, NULL ));
    }
    else if ( instr_per_loop == 24 ) {
        sum = ADD_VEC_SFP16(sum,test_hp_mac_VEC_FMA_24( iterations, EventSet, fp ));
        scalar_sum = ADD_VEC_SFP16(scalar_sum,test_hp_scalar_VEC_FMA_24( iterations, EventSet, NULL ));
    }
    else if ( instr_per_loop == 48 ) {
        sum = ADD_VEC_SFP16(sum,test_hp_mac_VEC_FMA_48( iterations, EventSet, fp ));
        scalar_sum = ADD_VEC_SFP16(scalar_sum,test_hp_scalar_VEC_FMA_48( iterations, EventSet, NULL ));
    }

    if( ((TEMP_TYPE*)&sum)[0] != ((TEMP_TYPE*)&scalar_sum)[0] ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}

#else
static
float test_hp_mac_VEC_FMA_12( uint64 iterations, int EventSet, FILE *fp ){

    (void)iterations;
    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(12, fp);
    }

    return 0.0;
}

static
float test_hp_mac_VEC_FMA_24( uint64 iterations, int EventSet, FILE *fp ){

    (void)iterations;
    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(24, fp);
    }

    return 0.0;
}

static
float test_hp_mac_VEC_FMA_48( uint64 iterations, int EventSet, FILE *fp ){

    (void)iterations;
    (void)EventSet;

    if ( NULL != fp ) {
      papi_stop_and_print_placeholder(48, fp);
    }

    return 0.0;
}

static
void test_hp_VEC_FMA( int instr_per_loop, uint64 iterations, int EventSet, FILE *fp )
{
    float sum = 0.0;
    float scalar_sum = 0.0;

    if ( instr_per_loop == 12 ) {
        sum += test_hp_mac_VEC_FMA_12( iterations, EventSet, fp );
        scalar_sum += test_hp_scalar_VEC_FMA_12( iterations, EventSet, NULL );
    }
    else if ( instr_per_loop == 24 ) {
        sum += test_hp_mac_VEC_FMA_24( iterations, EventSet, fp );
        scalar_sum += test_hp_scalar_VEC_FMA_24( iterations, EventSet, NULL );
    }
    else if ( instr_per_loop == 48 ) {
        sum += test_hp_mac_VEC_FMA_48( iterations, EventSet, fp );
        scalar_sum += test_hp_scalar_VEC_FMA_48( iterations, EventSet, NULL );
    }

    if( sum != scalar_sum ) {
        fprintf(stderr, "FMA: Inconsistent FLOP results detected!\n");
    }
}
#endif
