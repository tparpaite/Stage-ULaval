#!/bin/bash

dataset_list=('polynome' 'keijzer_6' 'nguyen_7' 'pagie_1' 'vladislavleva_4' 'boston' 'bioavailability' 'airfoil' 'onlinepop' 'compactiv' 'spacega')
dataset=${dataset_list[$1]}

echo "Debut GP classique ($dataset)"
python symbreg_deap.py $dataset
echo "Fin GP classique ($dataset)"

exit 0
