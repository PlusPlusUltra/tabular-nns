import openml
import numpy as np
import pandas as pd
import json
import os

from sklearn.model_selection import train_test_split

DATASETS={

"california_housing":537,
"cpu_act":562,
"Bike_Sharing_Demand":42712,
"abalone":183,
"wine_quality":287

}

BASE_PATH="data"

TEST_SIZE=0.2
VAL_SIZE=0.2

########################################
# HELPERS
########################################

def split_data(X,y,seed=42):

    X_train,X_test,y_train,y_test=train_test_split(

        X,y,
        test_size=TEST_SIZE,
        random_state=seed

    )

    X_train,X_val,y_train,y_val=train_test_split(

        X_train,y_train,
        test_size=VAL_SIZE,
        random_state=seed

    )

    return X_train,X_val,X_test,y_train,y_val,y_test


########################################
# MAIN LOOP
########################################

for name,oid in DATASETS.items():

    print("Downloading:",name)

    dataset=openml.datasets.get_dataset(oid)

    target=dataset.default_target_attribute

    X,y,cat_indicator,feature_names=dataset.get_data(
    
        target=target,
    
        dataset_format="dataframe"
    
    )

    ####################################
    # Detect numeric / categorical
    ####################################

    numeric_cols=[]
    cat_cols=[]

    for col,cat in zip(feature_names,cat_indicator):

        if cat:
            cat_cols.append(col)
        else:
            numeric_cols.append(col)

    X_num=X[numeric_cols] if numeric_cols else None

    X_cat=X[cat_cols] if cat_cols else None

    ####################################
    # Split
    ####################################

    X_train,X_val,X_test,y_train,y_val,y_test=split_data(X,y)

    ####################################
    # Extract numeric / cat splits
    ####################################

    if X_num is not None:

        N_train=X_train[numeric_cols].values
        N_val=X_val[numeric_cols].values
        N_test=X_test[numeric_cols].values

    if X_cat is not None:

        C_train=X_train[cat_cols].astype(str).values
        C_val=X_val[cat_cols].astype(str).values
        C_test=X_test[cat_cols].astype(str).values

    ####################################
    # Create folder
    ####################################

    path=f"{BASE_PATH}/{name}"

    os.makedirs(path,exist_ok=True)

    ####################################
    # Save numeric
    ####################################

    if X_num is not None:

        np.save(f"{path}/N_train.npy",N_train)
        np.save(f"{path}/N_val.npy",N_val)
        np.save(f"{path}/N_test.npy",N_test)

    ####################################
    # Save categorical
    ####################################

    if X_cat is not None:

        np.save(f"{path}/C_train.npy",C_train)
        np.save(f"{path}/C_val.npy",C_val)
        np.save(f"{path}/C_test.npy",C_test)

    ####################################
    # Save targets
    ####################################

    np.save(f"{path}/y_train.npy",y_train.values)
    np.save(f"{path}/y_val.npy",y_val.values)
    np.save(f"{path}/y_test.npy",y_test.values)

    ####################################
    # Info file
    ####################################

    info={

        "name":name,

        "n_num_features":len(numeric_cols),

        "n_cat_features":len(cat_cols),

        "train_size":len(y_train),

        "val_size":len(y_val),

        "test_size":len(y_test),

        "task_type":"regression",

        "openml_id":oid

    }

    with open(f"{path}/info.json","w") as f:

        json.dump(info,f,indent=4)

    print("Done:",name)

print("All datasets ready")