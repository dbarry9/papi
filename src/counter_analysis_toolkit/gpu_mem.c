#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <papi.h>
#include "gpu_mem.h"
#include "gpu_mem_kernels.h"

extern void gpu_mem(int EventSet, FILE *ofp_papi, unsigned int N, int tpb);

void gpu_mem_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir) {

    int sz, retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".gpu_mem";
    char *papiFileName;

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
    /* Invoke the memory kernels. */
    for( sz = 256; sz < 4*1024*1024; sz *= 2 ) {
        gpu_mem(EventSet, ofp_papi, 1.00*sz, hw_desc->warp_size);
        gpu_mem(EventSet, ofp_papi, 1.25*sz, hw_desc->warp_size);
        gpu_mem(EventSet, ofp_papi, 1.50*sz, hw_desc->warp_size);
        gpu_mem(EventSet, ofp_papi, 1.75*sz, hw_desc->warp_size);
    }
#else
    fprintf(stderr, "GPU Memory benchmark is not supported on this machine!\n");
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
