#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <papi.h>
#include "gpu_flops.h"
#include "gpu_flops_kernels.h"

extern void gpu_matrix_flop_dp(int EventSet, int N, FILE *ofp_papi, int type);
extern void gpu_matrix_flop_sp(int EventSet, int N, FILE *ofp_papi, int type);
extern void gpu_matrix_flop_hp(int EventSet, int N, FILE *ofp_papi, int type);

void gpu_flops_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir) {

    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".gpu_flops";
    char *papiFileName;

    (void)hw_desc;

    /* Create output file. */
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
 
    /* Create a PAPI event set. */ 
    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_add_named_event( EventSet, papi_event_name );
    if (retval != PAPI_OK ){
        fprintf(stderr, "Failed to add it to the set because error = %d\n", retval);
        goto error1;
    }

    /* Invoke the addition kernels. */
    gpu_matrix_flop_hp(EventSet, 16, ofp_papi, ADD);
    gpu_matrix_flop_hp(EventSet, 32, ofp_papi, ADD);
    gpu_matrix_flop_hp(EventSet, 64, ofp_papi, ADD);

    gpu_matrix_flop_sp(EventSet, 16, ofp_papi, ADD);
    gpu_matrix_flop_sp(EventSet, 32, ofp_papi, ADD);
    gpu_matrix_flop_sp(EventSet, 64, ofp_papi, ADD);

    gpu_matrix_flop_dp(EventSet, 16, ofp_papi, ADD);
    gpu_matrix_flop_dp(EventSet, 32, ofp_papi, ADD);
    gpu_matrix_flop_dp(EventSet, 64, ofp_papi, ADD);

    /* Invoke the subtraction kernels. */
    gpu_matrix_flop_hp(EventSet, 16, ofp_papi, SUB);
    gpu_matrix_flop_hp(EventSet, 32, ofp_papi, SUB);
    gpu_matrix_flop_hp(EventSet, 64, ofp_papi, SUB);

    gpu_matrix_flop_sp(EventSet, 16, ofp_papi, SUB);
    gpu_matrix_flop_sp(EventSet, 32, ofp_papi, SUB);
    gpu_matrix_flop_sp(EventSet, 64, ofp_papi, SUB);

    gpu_matrix_flop_dp(EventSet, 16, ofp_papi, SUB);
    gpu_matrix_flop_dp(EventSet, 32, ofp_papi, SUB);
    gpu_matrix_flop_dp(EventSet, 64, ofp_papi, SUB);

    /* Invoke the multiplication kernels. */
    gpu_matrix_flop_hp(EventSet, 16, ofp_papi, MUL);
    gpu_matrix_flop_hp(EventSet, 32, ofp_papi, MUL);
    gpu_matrix_flop_hp(EventSet, 64, ofp_papi, MUL);

    gpu_matrix_flop_sp(EventSet, 16, ofp_papi, MUL);
    gpu_matrix_flop_sp(EventSet, 32, ofp_papi, MUL);
    gpu_matrix_flop_sp(EventSet, 64, ofp_papi, MUL);

    gpu_matrix_flop_dp(EventSet, 16, ofp_papi, MUL);
    gpu_matrix_flop_dp(EventSet, 32, ofp_papi, MUL);
    gpu_matrix_flop_dp(EventSet, 64, ofp_papi, MUL);

    /* Invoke the division kernels. */
    gpu_matrix_flop_hp(EventSet, 16, ofp_papi, DIV);
    gpu_matrix_flop_hp(EventSet, 32, ofp_papi, DIV);
    gpu_matrix_flop_hp(EventSet, 64, ofp_papi, DIV);

    gpu_matrix_flop_sp(EventSet, 16, ofp_papi, DIV);
    gpu_matrix_flop_sp(EventSet, 32, ofp_papi, DIV);
    gpu_matrix_flop_sp(EventSet, 64, ofp_papi, DIV);

    gpu_matrix_flop_dp(EventSet, 16, ofp_papi, DIV);
    gpu_matrix_flop_dp(EventSet, 32, ofp_papi, DIV);
    gpu_matrix_flop_dp(EventSet, 64, ofp_papi, DIV);

    /* Invoke the SQRT kernels. */
    gpu_matrix_flop_hp(EventSet, 16, ofp_papi, SQRT);
    gpu_matrix_flop_hp(EventSet, 32, ofp_papi, SQRT);
    gpu_matrix_flop_hp(EventSet, 64, ofp_papi, SQRT);

    gpu_matrix_flop_sp(EventSet, 16, ofp_papi, SQRT);
    gpu_matrix_flop_sp(EventSet, 32, ofp_papi, SQRT);
    gpu_matrix_flop_sp(EventSet, 64, ofp_papi, SQRT);

    gpu_matrix_flop_dp(EventSet, 16, ofp_papi, SQRT);
    gpu_matrix_flop_dp(EventSet, 32, ofp_papi, SQRT);
    gpu_matrix_flop_dp(EventSet, 64, ofp_papi, SQRT);

    /* Invoke the FMA kernels. */
    gpu_matrix_flop_hp(EventSet, 16, ofp_papi, FMA);
    gpu_matrix_flop_hp(EventSet, 32, ofp_papi, FMA);
    gpu_matrix_flop_hp(EventSet, 64, ofp_papi, FMA);

    gpu_matrix_flop_sp(EventSet, 16, ofp_papi, FMA);
    gpu_matrix_flop_sp(EventSet, 32, ofp_papi, FMA);
    gpu_matrix_flop_sp(EventSet, 64, ofp_papi, FMA);

    gpu_matrix_flop_dp(EventSet, 16, ofp_papi, FMA);
    gpu_matrix_flop_dp(EventSet, 32, ofp_papi, FMA);
    gpu_matrix_flop_dp(EventSet, 64, ofp_papi, FMA);

    /* Invoke the MFMA kernels. */
    gpu_matrix_flop_sp(EventSet, 16,  ofp_papi, MFMA1);
    gpu_matrix_flop_dp(EventSet, 16,  ofp_papi, MFMA1);
    gpu_matrix_flop_sp(EventSet, 16,  ofp_papi, MFMA2);
    //gpu_matrix_flop_sp(EventSet, 16,  ofp_papi, MFMA3);
    //gpu_matrix_flop_hp(EventSet, 16,  ofp_papi, MFMA4);

    /* Clean-up. */
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
    return;
}
