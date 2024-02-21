#ifndef _GPU_FLOPS_
#define _GPU_FLOPS_

#include "hw_desc.h"

void gpu_flops_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir);
//void gpu_matrix_flop_dp(int EventSet, int N, FILE *ofp_papi, int type);
//void gpu_matrix_flop_sp(int EventSet, int N, FILE *ofp_papi, int type);
//void gpu_matrix_flop_hp(int EventSet, int N, FILE *ofp_papi, int type);

#endif
