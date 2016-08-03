#!/bin/bash

# Liste des jeux de donnees et creation des combinaisons d'appel possibles
dataset_list=('pagie_1' 'keijzer_6' 'nguyen_7')
dataset_fold_list=()

for i in ${dataset_list[@]};
do for j in `seq 0 4`;
    do dataset_fold_list+=("$i $j"); done;
done

# Combinaison dataset n_fold
dataset_fold=${dataset_fold_list[$1]}

# On se place dans le repertoire contenant la methode de regression symbolique
cd $HOME/Stage-ULaval/src/deap_tensorflow

echo "Debut GP coef ($dataset_fold)"
python symbreg_deap_tensorflow.py $dataset_fold
echo "Fin GP coef ($dataset_fold)"

exit 0
