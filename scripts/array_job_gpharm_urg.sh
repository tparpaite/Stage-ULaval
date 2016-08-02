#!/bin/bash

# Liste des jeux de donnees
dataset_list=('boston' 'bioavailability')
dataset=${dataset_list[$1]}

# On se place dans le repertoire contenant la methode de regression symbolique
cd $HOME/Stage-ULaval/src/symbreg_deap

echo "Debut GP harm ($dataset)"
python symbreg_deap_harm.py $dataset
echo "Fin GP harm ($dataset)"

exit 0
