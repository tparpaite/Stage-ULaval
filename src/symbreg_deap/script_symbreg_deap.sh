#!/bin/bash

rm -f logbook*

data_list="polynome boston airfoil onlinepop compactiv spacega"

for data in $data_list; do
    echo "Using classic genetic programming with dataset : $data..."
    python symbreg_deap.py $data
done

echo "Script finished, you can check logbook_mse to see results"

exit 0
