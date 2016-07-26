#!/bin/bash
#PBS -N GP classique
#PBS -A suj-571-aa
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:15:00
#PBS -t 1
#PBS -M tparpaite@gmail.com
#PBS -m bea
#PBS -o $HOME/Stage-ULaval/scripts/outputs/test_%I.out
#PBS -e $HOME/Stage-ULaval/scripts/outputs/test_%I.err

# Chargement des modules
module load compilers/gcc/4.9
module load libs/mkl/11.1
module load apps/python/2.7.10

# Lancement de l'execution (sur boston)
source $HOME/PYENV/bin/activate
cd $HOME/Stage-ULaval/scripts
./array_job_symbreg_gpclassic.sh 5 &

wait

exit 0
