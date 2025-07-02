#!/bin/bash

## Set to your PAPI installation.
PAPIDIR=${PAPIDIR}
export LD_LIBRARY_PATH=${PAPIDIR}/lib:${PWD}:${LD_LIBRARY_PATH}

## Use config file for EMR-specific architectural features.
cp emr.cat_cfg .cat_cfg

## Configure OpenMP for the multi-threaded data cache benchmarks.
export OMP_NUM_THREADS=4
#export OMP_PROC_BIND=spread
export OMP_PLACES="128,130,132,134"

## Run the CAT benchmarks.
for iter in {1..1}
do
    DIR=./output/${TYPE}/${iter}/
    rm -rf ${DIR}
    mkdir -p ${DIR}

    ./cat_collect -in ./input/branch_presets.txt -out ${DIR} -branch -instr -verbose
    ./cat_collect -in ./input/flops_presets.txt  -out ${DIR} -flops -vec -instr -verbose
    ./cat_collect -in ./input/dcache_presets.txt -out ${DIR} -dcr -dcw -instr -verbose
    ./cat_collect -in ./input/icache_presets.txt -out ${DIR} -ic -instr -verbose
    ./cat_collect -in ./input/tcache_presets.txt -out ${DIR} -dcr -dcw -ic -instr -verbose
    ./cat_collect -in ./input/instr_presets.txt  -out ${DIR} -instr -verbose
done
