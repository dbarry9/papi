#ifndef _GPU_FLOPS_
#define _GPU_FLOPS_

#include "hw_desc.h"

void gpu_flops_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir);

#endif
