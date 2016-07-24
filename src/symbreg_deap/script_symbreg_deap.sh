#!/bin/bash
#PBS -N GP classique
#PBS -A suj-571-aa
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -t [0-11]
#PBS -M tparpaite@gmail.com
#PBS -m bea

cd stage/Stage-ULaval/src/symbreg_deap

for i in $(seq 0 11)
do
    array_job_symbreg_deap.sh $MOAB_JOBARRAYINDEX &
done

wait
