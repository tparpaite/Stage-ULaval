#!/bin/bash
#PBS -N GP classique
#PBS -A suj-571-aa
#PBS -l nodes=1:ppn=8
#PBS -l walltime=12:00:00
#PBS -t [0-11]
#PBS -M tparpaite@gmail.com
#PBS -m bea

module load apps/python/2.7.10
source $HOME/ENV/bin/activate
cd $HOME/Stage-ULaval/scripts

./array_job_symbreg_deapharm.sh $MOAB_JOBARRAYINDEX &

wait
