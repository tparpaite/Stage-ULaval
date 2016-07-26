#!/bin/bash

# Liste des jeux de donnees
dataset_list=('polynome' 'keijzer_6' 'nguyen_7' 'pagie_1' 'vladislavleva_4' 'boston' 'bioavailability' 'airfoil' 'onlinepop' 'compactiv' 'spacega')
dataset=${dataset_list[$1]}

# On se place dans le repertoire contenant la methode de regression symbolique
cd $HOME/Stage-ULaval/src/sklearn/regression

echo "Debut regression lineaire ($dataset)"
python symbreg_linear.py $dataset
echo "Fin regression lineaire ($dataset)"

exit 0
