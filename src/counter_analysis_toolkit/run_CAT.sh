#!/bin/bash

## Set to your PAPI installation.
PAPIDIR=/home/dbarry/apps/papi-serotonin-zen5
export LD_LIBRARY_PATH=${PAPIDIR}/lib:${LD_LIBRARY_PATH}

## Configure OpenMP for the multi-threaded data cache benchmarks.
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread

## Run the CAT benchmarks.
for iter in {1..3}
do
    DIR=./output/${TYPE}/${iter}/
    mkdir -p ${DIR}

    ./cat_collect -in ./input/l3_native.txt -out ${DIR} -dcr -dcw -ic -instr -quick -verbose
done
