#!/bin/bash

# Liste des jeux de donnees
dataset_list=('polynome' 'keijzer_6' 'nguyen_7' 'pagie_1' 'vladislavleva_4' 'boston' 'bioavailability' 'airfoil' 'onlinepop' 'compactiv' 'spacega')
dataset=${dataset_list[$1]}

# On se place dans le repertoire contenant la methode
cd $HOME/Stage-ULaval/src/deap_tensorflow/tensorflow_huge_graph

echo "Debut mlp ($dataset)"
python tensorflow_huge_graph.py $dataset
echo "Fin mlp ($dataset)"

exit 0
