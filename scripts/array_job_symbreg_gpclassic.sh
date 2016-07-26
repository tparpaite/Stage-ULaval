#!/bin/bash

# Liste des jeux de donnees
dataset_list=('polynome' 'keijzer_6' 'nguyen_7' 'pagie_1' 'vladislavleva_4' 'boston' 'bioavailability' 'airfoil' 'onlinepop' 'compactiv' 'spacega')
dataset=${dataset_list[$1]}

# On se place dans le repertoire contenant la methode de regression symbolique
cd $HOME/Stage-ULaval/src/symbreg_deap

echo "Debut GP classique ($dataset)"
python symbreg_deap.py $dataset
echo "Fin GP classique ($dataset)"

exit 0
