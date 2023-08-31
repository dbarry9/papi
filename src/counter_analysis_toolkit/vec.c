#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <papi.h>
#include "vec.h"
#include "cat_arch.h"

int krnlIdx;
float **sigTable;

void vec_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir)
{
    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".vec";
    char *papiFileName;
    int i;
    krnlIdx = 0;
    sigTable = (float**)malloc(NUMKRNL*sizeof(float*));
    for( i = 0; i < NUMKRNL; ++i ) {
        sigTable[i] = (float*)malloc(NUMSIGS*sizeof(float));
    }

    sigTable[0][0] = 192;  sigTable[0][1] = 0;     sigTable[0][2] = 0;    sigTable[0][3] = 0;     sigTable[0][4] = 0;     sigTable[0][5] = 0;
    sigTable[1][0] = 384;  sigTable[1][1] = 0;     sigTable[1][2] = 0;    sigTable[1][3] = 0;     sigTable[1][4] = 0;     sigTable[1][5] = 0;
    sigTable[2][0] = 768;  sigTable[2][1] = 0;     sigTable[2][2] = 0;    sigTable[2][3] = 0;     sigTable[2][4] = 0;     sigTable[2][5] = 0;
    sigTable[3][0] = 384;  sigTable[3][1] = 0;     sigTable[3][2] = 0;    sigTable[3][3] = 0;     sigTable[3][4] = 0;     sigTable[3][5] = 0;
    sigTable[4][0] = 768;  sigTable[4][1] = 0;     sigTable[4][2] = 0;    sigTable[4][3] = 0;     sigTable[4][4] = 0;     sigTable[4][5] = 0;
    sigTable[5][0] = 1536; sigTable[5][1] = 0;     sigTable[5][2] = 0;    sigTable[5][3] = 0;     sigTable[5][4] = 0;     sigTable[5][5] = 0;
    sigTable[6][0] = 768;  sigTable[6][1] = 0;     sigTable[6][2] = 0;    sigTable[6][3] = 0;     sigTable[6][4] = 0;     sigTable[6][5] = 0;
    sigTable[7][0] = 1536; sigTable[7][1] = 0;     sigTable[7][2] = 0;    sigTable[7][3] = 0;     sigTable[7][4] = 0;     sigTable[7][5] = 0;
    sigTable[8][0] = 3072; sigTable[8][1] = 0;     sigTable[8][2] = 0;    sigTable[8][3] = 0;     sigTable[8][4] = 0;     sigTable[8][5] = 0;
    sigTable[9][0] = 0;    sigTable[9][1] = 96;    sigTable[9][2] = 0;    sigTable[9][3] = 0;     sigTable[9][4] = 0;     sigTable[9][5] = 0;
    sigTable[10][0] = 0;   sigTable[10][1] = 192;  sigTable[10][2] = 0;   sigTable[10][3] = 0;    sigTable[10][4] = 0;    sigTable[10][5] = 0;
    sigTable[11][0] = 0;   sigTable[11][1] = 384;  sigTable[11][2] = 0;   sigTable[11][3] = 0;    sigTable[11][4] = 0;    sigTable[11][5] = 0;
    sigTable[12][0] = 0;   sigTable[12][1] = 192;  sigTable[12][2] = 0;   sigTable[12][3] = 0;    sigTable[12][4] = 0;    sigTable[12][5] = 0;
    sigTable[13][0] = 0;   sigTable[13][1] = 384;  sigTable[13][2] = 0;   sigTable[13][3] = 0;    sigTable[13][4] = 0;    sigTable[13][5] = 0;
    sigTable[14][0] = 0;   sigTable[14][1] = 768;  sigTable[14][2] = 0;   sigTable[14][3] = 0;    sigTable[14][4] = 0;    sigTable[14][5] = 0;
    sigTable[15][0] = 0;   sigTable[15][1] = 384;  sigTable[15][2] = 0;   sigTable[15][3] = 0;    sigTable[15][4] = 0;    sigTable[15][5] = 0;
    sigTable[16][0] = 0;   sigTable[16][1] = 768;  sigTable[16][2] = 0;   sigTable[16][3] = 0;    sigTable[16][4] = 0;    sigTable[16][5] = 0;
    sigTable[17][0] = 0;   sigTable[17][1] = 1536; sigTable[17][2] = 0;   sigTable[17][3] = 0;    sigTable[17][4] = 0;    sigTable[17][5] = 0;
    sigTable[18][0] = 0;   sigTable[18][1] = 0;    sigTable[18][2] = 48;  sigTable[18][3] = 0;    sigTable[18][4] = 0;    sigTable[18][5] = 0;
    sigTable[19][0] = 0;   sigTable[19][1] = 0;    sigTable[19][2] = 96;  sigTable[19][3] = 0;    sigTable[19][4] = 0;    sigTable[19][5] = 0;
    sigTable[20][0] = 0;   sigTable[20][1] = 0;    sigTable[20][2] = 192; sigTable[20][3] = 0;    sigTable[20][4] = 0;    sigTable[20][5] = 0;
    sigTable[21][0] = 0;   sigTable[21][1] = 0;    sigTable[21][2] = 96;  sigTable[21][3] = 0;    sigTable[21][4] = 0;    sigTable[21][5] = 0;
    sigTable[22][0] = 0;   sigTable[22][1] = 0;    sigTable[22][2] = 192; sigTable[22][3] = 0;    sigTable[22][4] = 0;    sigTable[22][5] = 0;
    sigTable[23][0] = 0;   sigTable[23][1] = 0;    sigTable[23][2] = 384; sigTable[23][3] = 0;    sigTable[23][4] = 0;    sigTable[23][5] = 0;
    sigTable[24][0] = 0;   sigTable[24][1] = 0;    sigTable[24][2] = 192; sigTable[24][3] = 0;    sigTable[24][4] = 0;    sigTable[24][5] = 0;
    sigTable[25][0] = 0;   sigTable[25][1] = 0;    sigTable[25][2] = 384; sigTable[25][3] = 0;    sigTable[25][4] = 0;    sigTable[25][5] = 0;
    sigTable[26][0] = 0;   sigTable[26][1] = 0;    sigTable[26][2] = 768; sigTable[26][3] = 0;    sigTable[26][4] = 0;    sigTable[26][5] = 0;
    sigTable[27][0] = 0;   sigTable[27][1] = 0;    sigTable[27][2] = 0;   sigTable[27][3] = 192;  sigTable[27][4] = 0;    sigTable[27][5] = 0;
    sigTable[28][0] = 0;   sigTable[28][1] = 0;    sigTable[28][2] = 0;   sigTable[28][3] = 384;  sigTable[28][4] = 0;    sigTable[28][5] = 0;
    sigTable[29][0] = 0;   sigTable[29][1] = 0;    sigTable[29][2] = 0;   sigTable[29][3] = 768;  sigTable[29][4] = 0;    sigTable[29][5] = 0;
    sigTable[30][0] = 0;   sigTable[30][1] = 0;    sigTable[30][2] = 0;   sigTable[30][3] = 384;  sigTable[30][4] = 0;    sigTable[30][5] = 0;
    sigTable[31][0] = 0;   sigTable[31][1] = 0;    sigTable[31][2] = 0;   sigTable[31][3] = 768;  sigTable[31][4] = 0;    sigTable[31][5] = 0;
    sigTable[32][0] = 0;   sigTable[32][1] = 0;    sigTable[32][2] = 0;   sigTable[32][3] = 1536; sigTable[32][4] = 0;    sigTable[32][5] = 0;
    sigTable[33][0] = 0;   sigTable[33][1] = 0;    sigTable[33][2] = 0;   sigTable[33][3] = 768;  sigTable[33][4] = 0;    sigTable[33][5] = 0;
    sigTable[34][0] = 0;   sigTable[34][1] = 0;    sigTable[34][2] = 0;   sigTable[34][3] = 1536; sigTable[34][4] = 0;    sigTable[34][5] = 0;
    sigTable[35][0] = 0;   sigTable[35][1] = 0;    sigTable[35][2] = 0;   sigTable[35][3] = 3072; sigTable[35][4] = 0;    sigTable[35][5] = 0;
    sigTable[36][0] = 0;   sigTable[36][1] = 0;    sigTable[36][2] = 0;   sigTable[36][3] = 0;    sigTable[36][4] = 96;   sigTable[36][5] = 0;
    sigTable[37][0] = 0;   sigTable[37][1] = 0;    sigTable[37][2] = 0;   sigTable[37][3] = 0;    sigTable[37][4] = 192;  sigTable[37][5] = 0;
    sigTable[38][0] = 0;   sigTable[38][1] = 0;    sigTable[38][2] = 0;   sigTable[38][3] = 0;    sigTable[38][4] = 384;  sigTable[38][5] = 0;
    sigTable[39][0] = 0;   sigTable[39][1] = 0;    sigTable[39][2] = 0;   sigTable[39][3] = 0;    sigTable[39][4] = 192;  sigTable[39][5] = 0;
    sigTable[40][0] = 0;   sigTable[40][1] = 0;    sigTable[40][2] = 0;   sigTable[40][3] = 0;    sigTable[40][4] = 384;  sigTable[40][5] = 0;
    sigTable[41][0] = 0;   sigTable[41][1] = 0;    sigTable[41][2] = 0;   sigTable[41][3] = 0;    sigTable[41][4] = 768;  sigTable[41][5] = 0;
    sigTable[42][0] = 0;   sigTable[42][1] = 0;    sigTable[42][2] = 0;   sigTable[42][3] = 0;    sigTable[42][4] = 384;  sigTable[42][5] = 0;
    sigTable[43][0] = 0;   sigTable[43][1] = 0;    sigTable[43][2] = 0;   sigTable[43][3] = 0;    sigTable[43][4] = 768;  sigTable[43][5] = 0;
    sigTable[44][0] = 0;   sigTable[44][1] = 0;    sigTable[44][2] = 0;   sigTable[44][3] = 0;    sigTable[44][4] = 1536; sigTable[44][5] = 0;
    sigTable[45][0] = 0;   sigTable[45][1] = 0;    sigTable[45][2] = 0;   sigTable[45][3] = 0;    sigTable[45][4] = 0;    sigTable[45][5] = 48;
    sigTable[46][0] = 0;   sigTable[46][1] = 0;    sigTable[46][2] = 0;   sigTable[46][3] = 0;    sigTable[46][4] = 0;    sigTable[46][5] = 96;
    sigTable[47][0] = 0;   sigTable[47][1] = 0;    sigTable[47][2] = 0;   sigTable[47][3] = 0;    sigTable[47][4] = 0;    sigTable[47][5] = 192;
    sigTable[48][0] = 0;   sigTable[48][1] = 0;    sigTable[48][2] = 0;   sigTable[48][3] = 0;    sigTable[48][4] = 0;    sigTable[48][5] = 96;
    sigTable[49][0] = 0;   sigTable[49][1] = 0;    sigTable[49][2] = 0;   sigTable[49][3] = 0;    sigTable[49][4] = 0;    sigTable[49][5] = 192;
    sigTable[50][0] = 0;   sigTable[50][1] = 0;    sigTable[50][2] = 0;   sigTable[50][3] = 0;    sigTable[50][4] = 0;    sigTable[50][5] = 384;
    sigTable[51][0] = 0;   sigTable[51][1] = 0;    sigTable[51][2] = 0;   sigTable[51][3] = 0;    sigTable[51][4] = 0;    sigTable[51][5] = 192;
    sigTable[52][0] = 0;   sigTable[52][1] = 0;    sigTable[52][2] = 0;   sigTable[52][3] = 0;    sigTable[52][4] = 0;    sigTable[52][5] = 384;
    sigTable[53][0] = 0;   sigTable[53][1] = 0;    sigTable[53][2] = 0;   sigTable[53][3] = 0;    sigTable[53][4] = 0;    sigTable[53][5] = 768;

    (void)hw_desc;

    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    if (NULL == (papiFileName = (char *)calloc( 1+l, sizeof(char)))) {
        return;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        goto error0;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Failed to open file %s.\n", papiFileName);
        goto error0;
    }

    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_add_named_event( EventSet, papi_event_name );
    if (retval != PAPI_OK ){
        goto error1;
    }

    /* Create output file header. */
    fprintf(ofp_papi, "#N RawEvtCnt HP_Ops_Sep SP_Ops_Sep DP_Ops_Sep HP_Ops_FMA SP_Ops_FMA DP_Ops_FMA\n");

#if defined(X86)

#if defined(AVX128_AVAIL)

    // Non-FMA instruction trials.
    test_hp_x86_128B_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC( 96, ITER, EventSet, ofp_papi );

#if defined(AVX256_AVAIL)
    test_hp_x86_256B_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC( 96, ITER, EventSet, ofp_papi );

#if defined(AVX512_AVAIL)
    test_hp_x86_512B_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC( 96, ITER, EventSet, ofp_papi );
#endif
#endif

    test_sp_x86_128B_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC( 96, ITER, EventSet, ofp_papi );

#if defined(AVX256_AVAIL)
    test_sp_x86_256B_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC( 96, ITER, EventSet, ofp_papi );

#if defined(AVX512_AVAIL)
    test_sp_x86_512B_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC( 96, ITER, EventSet, ofp_papi );
#endif
#endif

    test_dp_x86_128B_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC( 96, ITER, EventSet, ofp_papi );

#if defined(AVX256_AVAIL)
    test_dp_x86_256B_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC( 96, ITER, EventSet, ofp_papi );

#if defined(AVX512_AVAIL)
    test_dp_x86_512B_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC( 96, ITER, EventSet, ofp_papi );
#endif
#endif

    // FMA instruction trials.
    test_hp_x86_128B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_128B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#if defined(AVX256_AVAIL)
    test_hp_x86_256B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_256B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#if defined(AVX512_AVAIL)
    test_hp_x86_512B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_x86_512B_VEC_FMA( 48, ITER, EventSet, ofp_papi );
#endif
#endif

    test_sp_x86_128B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_128B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#if defined(AVX256_AVAIL)
    test_sp_x86_256B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_256B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#if defined(AVX512_AVAIL)
    test_sp_x86_512B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_x86_512B_VEC_FMA( 48, ITER, EventSet, ofp_papi );
#endif
#endif

    test_dp_x86_128B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_128B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#if defined(AVX256_AVAIL)
    test_dp_x86_256B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_256B_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#if defined(AVX512_AVAIL)
    test_dp_x86_512B_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_x86_512B_VEC_FMA( 48, ITER, EventSet, ofp_papi );
#endif
#endif

#else
    fprintf(stderr, "Vector FLOP benchmark is not supported on this architecture: AVX unavailable!\n");
#endif

#elif defined(ARM)

    // Non-FMA instruction trials.
    test_hp_arm_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC( 96, ITER, EventSet, ofp_papi );

    test_sp_arm_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC( 96, ITER, EventSet, ofp_papi );

    test_dp_arm_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC( 96, ITER, EventSet, ofp_papi );

    // FMA instruction trials.
    test_hp_arm_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_arm_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    test_sp_arm_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_arm_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    test_dp_arm_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_arm_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#elif defined(POWER)

    // Non-FMA instruction trials.
    test_hp_power_VEC( 24, ITER, EventSet, ofp_papi );
    test_hp_power_VEC( 48, ITER, EventSet, ofp_papi );
    test_hp_power_VEC( 96, ITER, EventSet, ofp_papi );

    test_sp_power_VEC( 24, ITER, EventSet, ofp_papi );
    test_sp_power_VEC( 48, ITER, EventSet, ofp_papi );
    test_sp_power_VEC( 96, ITER, EventSet, ofp_papi );

    test_dp_power_VEC( 24, ITER, EventSet, ofp_papi );
    test_dp_power_VEC( 48, ITER, EventSet, ofp_papi );
    test_dp_power_VEC( 96, ITER, EventSet, ofp_papi );

    // FMA instruction trials.
    test_hp_power_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_hp_power_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_hp_power_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    test_sp_power_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_sp_power_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_sp_power_VEC_FMA( 48, ITER, EventSet, ofp_papi );

    test_dp_power_VEC_FMA( 12, ITER, EventSet, ofp_papi );
    test_dp_power_VEC_FMA( 24, ITER, EventSet, ofp_papi );
    test_dp_power_VEC_FMA( 48, ITER, EventSet, ofp_papi );

#endif

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }
    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

error1:
    fclose(ofp_papi);
error0:
    free(papiFileName);
    for( i = 0; i < NUMKRNL; ++i ) {
        free(sigTable[i]);
    }
    free(sigTable);
    return;
}
