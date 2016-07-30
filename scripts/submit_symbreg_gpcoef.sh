#!/bin/bash
#PBS -N GP coef
#PBS -A suj-571-aa
#PBS -l nodes=1:gpus=6
#PBS -l walltime=12:00:00
#PBS -t [0-14]
#PBS -M tparpaite@gmail.com
#PBS -m bea
#PBS -o $HOME/Stage-ULaval/scripts/outputs/gpcoef_%I.out
#PBS -e $HOME/Stage-ULaval/scripts/outputs/gpcoef_%I.err

# Chargement des modules
module load compilers/gcc/4.8.5
module load cuda/7.5.18
module load libs/cuDNN/5
module load libs/mkl/11.1
module load apps/python/2.7.10

# Lancement de l'execution
source $HOME/TFENV/bin/activate
cd $HOME/Stage-ULaval/scripts
./array_job_symbreg_gpcoef.sh $MOAB_JOBARRAYINDEX &

wait

exit 0
