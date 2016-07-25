#!/bin/bash

dataset_list=('polynome' 'keijzer_6' 'nguyen_7' 'pagie_1' 'vladislavleva_4' 'boston' 'bioavailability' 'airfoil' 'onlinepop' 'compactiv' 'spacega')
dataset=${dataset_list[$1]}

cd $HOME/Stage-ULaval/src/symbreg_deap

echo "Debut GP harm ($dataset)"
python symbreg_deap_harm.py $dataset
echo "Fin GP harm ($dataset)"

exit 0
