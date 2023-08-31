#ifndef _VEC_
#define _VEC_

#include "hw_desc.h"

#define ITER 1
#define INNER_ITER 1
#define NUMKRNL 54
#define NUMSIGS 6

void vec_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir);

#endif
