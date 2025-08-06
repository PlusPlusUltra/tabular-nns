
from tqdm import tqdm
from TALENT.model.utils import get_deep_args,show_results,tune_hyper_parameters,get_method,set_seeds
from TALENT.model.lib.data import get_dataset

import os
#from your_outlier_file import detect_outliers_statistical, detect_outliers_sklearn  # or inline logic

def log_misclassified_outliers(dataset_name, X_test, y_test, y_pred):
    from sklearn.metrics import accuracy_score
    misclassified_mask = y_test != y_pred
    X_misclassified = X_test[misclassified_mask]

    methods = ["ZScore", "ModifiedZScore", "IQR", "IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
    result_lines = []
""""
    for method in methods:
        try:
            outlier_mask = detect_outliers_statistical(X_test, method) if method in ["ZScore", "ModifiedZScore", "IQR"] \
                else detect_outliers_sklearn(X_test, method)
            outlier_mask = np.asarray(outlier_mask)
            misclassified_outliers = (misclassified_mask & (~outlier_mask)).sum()
            result_lines.append(f"{method}: {misclassified_outliers} out of {misclassified_mask.sum()} misclassified")
        except Exception as e:
            result_lines.append(f"{method}: ERROR ({str(e)})")

    # Save to log file
    os.makedirs("misclassified_outliers", exist_ok=True)
    log_path = os.path.join("misclassified_outliers", f"{dataset_name}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(result_lines))

"""

if __name__ == '__main__':
    loss_list, results_list, time_list = [], [], []
    args,default_para,opt_space = get_deep_args()
    train_val_data, test_data, info = get_dataset(
    args.dataset,
    args.dataset_path,
    remove_outliers=args.remove_outliers,
    outlier_method=args.outlier_method
)

    if args.tune:
        args = tune_hyper_parameters(args,opt_space,train_val_data,info)
    for seed in tqdm(range(args.seed_num)):
        args.seed = seed    # update seed  
        set_seeds(args.seed)
        method = get_method(args.model_type)(args, info['task_type'] == 'regression')
        time_cost = method.fit(train_val_data, info)    
        vl, vres, metric_name, predict_logits = method.predict(test_data, info, model_name=args.evaluate_option)
        loss_list.append(vl)
        results_list.append(vres)
        time_list.append(time_cost)

    show_results(args,info, metric_name,loss_list,results_list,time_list)
