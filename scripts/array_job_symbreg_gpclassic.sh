#!/bin/bash

# Liste des jeux de donnees
dataset_list=('keijzer_6' 'boston' 'bioavailability')
dataset=${dataset_list[$1]}

# On se place dans le repertoire contenant la methode de regression symbolique
cd $HOME/Stage-ULaval/src/symbreg_deap

echo "Debut GP classique ($dataset)"
python symbreg_deap.py $dataset
echo "Fin GP classique ($dataset)"

exit 0
