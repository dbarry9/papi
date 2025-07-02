#!/bin/bash

## Set to your PAPI installation.
PAPIDIR=${PAPIDIR}
export LD_LIBRARY_PATH=${PAPIDIR}/lib:${LD_LIBRARY_PATH}

make realclean
make PAPI_DIR=${PAPIDIR} ARCH=X86
