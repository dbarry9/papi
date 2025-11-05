#!/bin/bash

## Set to your PAPI installation.
PAPIDIR=${PAPI_DIR}
export LD_LIBRARY_PATH=${PAPIDIR}/lib:${PWD}:${LD_LIBRARY_PATH}

## Use config file for EMR-specific architectural features.
cp emr.cat_cfg .cat_cfg

## Configure OpenMP for the multi-threaded data cache benchmarks.
export OMP_NUM_THREADS=1
export OMP_PLACES="128"
#export OMP_PROC_BIND=spread

## Run the CAT benchmarks.
for iter in {1..1}
do
    DIR=./output/${TYPE}/${iter}/
    mkdir -p ${DIR}

    ./cat_collect -in ./input/dcache_presets_1.txt -out ${DIR} -dcr -dcw -instr
    ./cat_collect -in ./input/dcache_presets_2.txt -out ${DIR} -dcr -dcw -instr
    ./cat_collect -in ./input/dcache_presets_3.txt -out ${DIR} -dcr -dcw -instr
done
