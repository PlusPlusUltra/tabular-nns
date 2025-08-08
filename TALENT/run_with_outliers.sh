#!/bin/bash

#models=("mlp" "RandomForest" "xgboost" "LinearRegression")
models=("mlp" "xgboost")
datasets=("Diamonds" "2dplanes" "1000-Cameras-Dataset" "Abalone_reg" "Brazillian_houses_reproduced" "Data_Science_Salaries")
methods=("IsolationForest" "LocalOutlierFactor" "OneClassSVM" "ZScore" "ModifiedZScore" "IQR")

output_file="results_with_outliers.txt"
touch "$output_file" 

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            # Checking if an experiment is already done
            if grep -q "^$dataset,$model,$method," "$output_file"; then
                echo "Skipping $dataset,$model,$method — already done"
                continue
            fi

            echo "Running model=$model, dataset=$dataset, method=$method"

            if [ "$model" == "mlp" ]; then
                script="train_model_deep.py"
            else
                script="train_model_classical.py"
            fi

            output=$(python "$script" \
                --model_type "$model" \
                --dataset "$dataset" \
                --remove_outliers \
                --outlier_method "$method")

            value=$(echo "$output" | grep -E "Accuracy MEAN =|RMSE MEAN =" | awk '{print $4}')

            if [ -n "$value" ]; then
                echo "$dataset,$model,$method,$value" >> "$output_file"
            else
                echo "$dataset,$model,$method,ERROR" >> "$output_file"
            fi
        done
    done
done

echo "All outlier experiments complete (or skipped). Results saved to $output_file"
