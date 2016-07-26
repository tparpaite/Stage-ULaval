#!/bin/bash
#PBS -N GP classique
#PBS -A suj-571-aa
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -t [0-11]
#PBS -M tparpaite@gmail.com
#PBS -m bea
#PBS -o $HOME/Stage-ULaval/scripts/ouputs/gpcoef_%I.out
#PBS -e $HOME/Stage-ULaval/scripts/ouputs/gpcoef_%I.err

# Chargement des modules
module load compilers/gcc/4.9
module load libs/mkl/11.1
module load apps/python/2.7.10
# SUR HELIOS ?
# TODO : CHARGER TENSORFLOW + LIB CUDA

# Lancement de l'execution
source $HOME/PYENV/bin/activate
cd $HOME/Stage-ULaval/scripts
./array_job_symbreg_gpcoef.sh $MOAB_JOBARRAYINDEX &

wait

exit 0
