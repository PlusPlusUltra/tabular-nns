#!/bin/bash

# Define models and datasets
models=("mlp" "RandomForest" "xgboost" "LinearRegression")
datasets=("credit-g" "electricity" "elevators" "socmob" "splice" "vehicle" "kc1" "phoneme" "pol" "house_16h" "eye_movements" "cmc" "connect-4" "eucalyptus")

# Output file
output_file="results.txt"
touch "$output_file"  # Ensure the file exists

# Loop through each model and dataset combination
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        # Check if this combination is already in the results file
        if grep -q "^$dataset,$model," "$output_file"; then
            echo "Skipping $dataset,$model — already done"
            continue
        fi

        echo "Running model=$model on dataset=$dataset"

        # Choose the correct script
        if [ "$model" == "mlp" ]; then
            script="train_model_deep.py"
        else
            script="train_model_classical.py"
        fi

        # Run the command and capture output
        output=$(python "$script" --model_type "$model" --dataset "$dataset")

        # Extract either Accuracy MEAN or RMSE MEAN
        value=$(echo "$output" | grep -E "Accuracy MEAN =|RMSE MEAN =" | awk '{print $4}')

        # Write result
        if [ -n "$value" ]; then
            echo "$dataset,$model,$value" >> "$output_file"
        else
            echo "$dataset,$model,ERROR" >> "$output_file"
        fi
    done
done

echo "All experiments complete (or skipped if already done). Results saved to $output_file"
