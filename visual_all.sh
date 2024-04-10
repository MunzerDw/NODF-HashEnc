#!/bin/bash

# Define the base directory containing the data
base_dir="output/paper_results/data_stanford"

# Define the different model directories to iterate over
models=(
  "output/paper_results/data_stanford/ours_14lvls20hm_1embedsize" 
  "output/paper_results/data_stanford/ours_14lvls20hm_4embedsize" 
)

# And over each model
for model in "${models[@]}"; do
  # Construct the predictions_path dynamically
  out_folder="${model}"
  
  # Check if the file exists before attempting to run the command
  echo "============================================================="
  echo "===================== $model ====================="
  echo "============================================================="
  # Run the evaluation command
  python visualize.py --out_folder "$out_folder"
  echo ""
  echo ""
  echo ""
done
