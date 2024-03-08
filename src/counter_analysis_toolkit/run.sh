#!/bin/bash

# Load modules.
module use e4s
## upon sourcing many more modules become available
source /packages/e4s/23.11/mvapich/spack/share/spack/setup-env.sh

module load cuda/12.2.2-gcc-12.2.1-l2tf
module load openmpi/4.1.6-gcc-12.2.1-jypz
module load git/2.42.0-gcc-12.2.1-t4jd

export PAPI_CUDA_ROOT=/packages/cuda/12.2.2

PAPIDIR=${HOME}/apps/papi-hopper
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PAPIDIR}/lib

OUTPUT=${HOME}/src/papi/src/counter_analysis_toolkit/output
mkdir -p ${OUTPUT}

#${PAPIDIR}/bin/papi_command_line "cuda:::ctc__rx_bytes_data_user:device=0"
#exit

#./cat_collect -in ./arm_flops.txt -out ${OUTPUT}/ -vec -flops
#for iter in {1..5}
#do
#    MYDIR=${OUTPUT}/branch/${iter}
#    mkdir -p ${MYDIR}
#    ./cat_collect -in ./wholelist.txt -out ${MYDIR}/ -branch
#done

#for rep in {1..1}
#for rep in {2..2}
#for rep in {3..3}
#for rep in {4..4}
for rep in {5..5}
do
    mkdir -p ${OUTPUT}/cat/${rep}
    ./cat_collect -in cuda.txt -out ${OUTPUT}/cat/${rep}/ -gpu_flops
done
