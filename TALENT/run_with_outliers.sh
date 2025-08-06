#!/bin/bash

# Define models, datasets, and outlier methods
models=("mlp" "RandomForest" "xgboost" "LinearRegression")
datasets=("credit-g" "electricity" "elevators" "socmob" "splice" "vehicle" "kc1" "phoneme" "pol" "house_16h" "eye_movements" "cmc" "connect-4" "eucalyptus")
methods=("IsolationForest" "LocalOutlierFactor" "OneClassSVM" "ZScore" "ModifiedZScore" "IQR")

# Output file
output_file="results_with_outliers.txt"
touch "$output_file"  # Ensure the file exists

# Loop through all combinations
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            # Check if this combination is already in the results file
            if grep -q "^$dataset,$model,$method," "$output_file"; then
                echo "Skipping $dataset,$model,$method — already done"
                continue
            fi

            echo "Running model=$model, dataset=$dataset, method=$method"

            # Choose the correct script
            if [ "$model" == "mlp" ]; then
                script="train_model_deep.py"
            else
                script="train_model_classical.py"
            fi

            # Run the command and capture output
            output=$(python "$script" \
                --model_type "$model" \
                --dataset "$dataset" \
                --remove_outliers \
                --outlier_method "$method")

            # Extract either Accuracy MEAN or RMSE MEAN
            value=$(echo "$output" | grep -E "Accuracy MEAN =|RMSE MEAN =" | awk '{print $4}')

            # Write result
            if [ -n "$value" ]; then
                echo "$dataset,$model,$method,$value" >> "$output_file"
            else
                echo "$dataset,$model,$method,ERROR" >> "$output_file"
            fi
        done
    done
done

echo "All outlier experiments complete (or skipped). Results saved to $output_file"
