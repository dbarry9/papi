#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <papi.h>
#include "cat_gpu_arch.h"
#include "gpu_flops.h"

extern void gpu_matrix_flop_dp(int EventSet, FILE *ofp_papi);
extern void gpu_matrix_flop_sp(int EventSet, FILE *ofp_papi);
extern void gpu_matrix_flop_hp(int EventSet, FILE *ofp_papi);

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

#if defined(GPU_AMD)
    /* Invoke the addition kernels. */
    gpu_matrix_flop_hp(EventSet, ofp_papi);
    gpu_matrix_flop_sp(EventSet, ofp_papi);
    gpu_matrix_flop_dp(EventSet, ofp_papi);

    /* Invoke the MFMA kernels. */
    //gpu_matrix_flop_sp(EventSet, 16,  ofp_papi, MFMA1);
    //gpu_matrix_flop_dp(EventSet, 16,  ofp_papi, MFMA1);
    //gpu_matrix_flop_sp(EventSet, 16,  ofp_papi, MFMA2);
    //gpu_matrix_flop_sp(EventSet, 16,  ofp_papi, MFMA3);
    //gpu_matrix_flop_hp(EventSet, 16,  ofp_papi, MFMA4);

#elif defined(GPU_NVIDIA)
    gpu_matrix_flop_hp(EventSet, ofp_papi);
    gpu_matrix_flop_sp(EventSet, ofp_papi);
    gpu_matrix_flop_dp(EventSet, ofp_papi);

#else
    fprintf(stderr, "GPU FLOPs benchmark is not supported on this machine!\n");
#endif

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
