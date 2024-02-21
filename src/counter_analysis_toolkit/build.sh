#!/bin/bash

PAPIDIR=${HOME}/apps/papi-hopper
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PAPIDIR}/lib

#module load gcc/13.2
##module load e4s/23.11/mpich
#module load gcc/12.2.1
#module load cuda/12.3.2
#module load e4s/23.11/mvapich

module use e4s
## upon sourcing many more modules become available
source /packages/e4s/23.11/mvapich/spack/share/spack/setup-env.sh

module load cuda/12.2.2-gcc-12.2.1-l2tf
module load openmpi/4.1.6-gcc-12.2.1-jypz
module load git/2.42.0-gcc-12.2.1-t4jd

gcc -v
echo "CC=${CC}"

OUTPUT=${HOME}/src/papi/src/counter_analysis_toolkit/output
mkdir -p ${OUTPUT}

make realclean

clear

#make PAPIDIR=${PAPIDIR} USEMPI=false CPU_ARCH=ARM
make PAPIDIR=${PAPIDIR} USEMPI=false CPU_ARCH=ARM GPU_ARCH=NVIDIA CUDADIR=/packages/cuda/12.2.2
