#!/bin/bash

## Set to your PAPI installation.
PAPIDIR=/home/dbarry/apps/papi-serotonin-zen5
export LD_LIBRARY_PATH=${PAPIDIR}/lib:${LD_LIBRARY_PATH}

## Use config file for Zen5-specific architectural features.
cp zen5.cat_cfg .cat_cfg

## Configure OpenMP for the multi-threaded data cache benchmarks.
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread

## Run the CAT benchmarks.
for iter in {1..3}
do
    DIR=./output/${TYPE}/${iter}/
    rm -rf ${DIR}
    mkdir -p ${DIR}

    ./cat_collect -in ./input/branch_presets.txt -out ${DIR} -branch -instr -verbose
    ./cat_collect -in ./input/flops_presets.txt  -out ${DIR} -flops -vec -instr -verbose
    ./cat_collect -in ./input/dcache_presets.txt -out ${DIR} -dcr -dcw -instr
    ./cat_collect -in ./input/icache_presets.txt -out ${DIR} -ic -instr -verbose
    ./cat_collect -in ./input/tcache_presets.txt -out ${DIR} -dcr -dcw -ic -instr -verbose
    ./cat_collect -in ./input/instr_presets.txt  -out ${DIR} -instr -verbose
done
