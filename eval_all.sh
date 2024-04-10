#!/bin/bash

# Define the base directory containing the data
base_dir="output/paper_results/data_stanford"

# Define the different model directories to iterate over
models=("ours" "baseline")

# Define the different M values
M_values=(70 40 20)

# Iterate over each M value
for M in "${M_values[@]}"; do
  # And over each model
  for model in "${models[@]}"; do
    # Construct the predictions_path dynamically
    predictions_path="${base_dir}/${model}_M${M}/pointwise_estimates.pt"
    out_folder="${base_dir}/${model}_M${M}"
    
    # Check if the file exists before attempting to run the command
    echo "============================================================="
    echo "===================== $model with M=$M ====================="
    echo "============================================================="
    if [[ -f "$predictions_path" ]]; then
      # Run the evaluation command
      python evaluate.py --device cpu --predictions_path "$predictions_path" --M "$M" --out_folder "$out_folder"
    else
      echo "File not found: $predictions_path"
    echo ""
    echo ""
    echo ""
    fi
  done
done
