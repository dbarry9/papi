#!/bin/bash

## Set to your PAPI installation.
PAPIDIR=${PAPIDIR}
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
    rm -rf ${DIR}
    mkdir -p ${DIR}

    ./cat_collect -in ./input/dcache_presets_1.txt -out ${DIR} -dcr -dcw -instr
    ./cat_collect -in ./input/branch_presets.txt -out ${DIR} -branch -instr -verbose
    ./cat_collect -in ./input/flops_presets.txt  -out ${DIR} -flops -vec -instr -verbose
    ./cat_collect -in ./input/dcache_presets_2.txt -out ${DIR} -dcr -dcw -instr
    ./cat_collect -in ./input/icache_presets.txt -out ${DIR} -ic -instr -verbose
    ./cat_collect -in ./input/dcache_presets_3.txt -out ${DIR} -dcr -dcw -instr
    ./cat_collect -in ./input/instr_presets.txt  -out ${DIR} -instr -verbose

    ${PAPIDIR}/bin/papi_command_line "PAPI_L1_TCM"  > ${DIR}/tcache_command_line.txt
    ${PAPIDIR}/bin/papi_command_line "PAPI_L2_TCA" >> ${DIR}/tcache_command_line.txt
    ${PAPIDIR}/bin/papi_command_line "PAPI_L2_TCM" >> ${DIR}/tcache_command_line.txt
    ${PAPIDIR}/bin/papi_command_line "PAPI_L2_TCR" >> ${DIR}/tcache_command_line.txt
    ${PAPIDIR}/bin/papi_command_line "PAPI_L3_TCA" >> ${DIR}/tcache_command_line.txt
    ${PAPIDIR}/bin/papi_command_line "PAPI_L3_TCM" >> ${DIR}/tcache_command_line.txt

done
