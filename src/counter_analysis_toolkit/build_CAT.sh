#!/bin/bash

PAPIDIR=/home/dbarry/apps/papi-serotonin-zen5
export LD_LIBRARY_PATH=${PAPIDIR}/lib:${LD_LIBRARY_PATH}

make realclean
make PAPI_DIR=${PAPIDIR} ARCH=X86
