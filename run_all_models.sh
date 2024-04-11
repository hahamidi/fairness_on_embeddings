#!/bin/bash

# Specify the directory containing the .sh files
directory="./sbatch/test/*.sh"

# Iterate over each .sh file in the directory
for file in $directory; do

    echo "Submitted $file"
    sbatch "$file"

done
