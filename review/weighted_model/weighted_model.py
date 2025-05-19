from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter 
import os
import sys
import json 

df = pd.read_csv("whole_2006_2023_data_weight.csv")

#Select train data which is 80% of the entire dataset
#This was precalculated
train_df = df[:41034]
epsilon = 1e-6
train_df ['weight'] = df['weight'].clip(lower=epsilon)
#Transform weights by taking square root to compress range
train_df['weight']=np.sqrt(train_df['weight'])

#Transform weights by taking log to compress range
#train_df['weight']=np.log1p(train_df['weight'])

weights = train_df['weight'].values
test_df = df[41034:]

path=os.getcwd()
#print(path)
print("LOADING DATA, WAIT.......\n\n\n")

#Train features and matrices
X = train_df.iloc[:,3:] #Training
y = train_df.iloc[:,2:3]
#print("Training data\n\n\n\n", X_train)
#print("Training labels\n\n\n\n", y_train)

#Test features and matrices
X_test = test_df.iloc[:,3:]
X_test_raw = test_df.iloc[:,3:]
y_test = test_df.iloc[:,2:3]
y_test_raw = test_df.iloc[:,2:3]

#define features
features = train_df.columns
myfeatures = list(features[3:])

#DMatrices
dtrain = xgb.DMatrix(X, label=y, weight=weights,  missing=-999.0, feature_names=myfeatures) # = xgb.DMatrix(y_train)
#dtrain = xgb.DMatrix(X, label=y,  missing=-999.0, feature_names=myfeatures) # = xgb.DMatrix(y_train)
#Validation set
dtest = xgb.DMatrix(X_test, label=y_test, missing=-999.0, feature_names=myfeatures) # = xgb.DMatrix(y_test)

X_test = xgb.DMatrix(X_test)
y_test = xgb.DMatrix(y_test)
print("Data Preparation succesful.....\n")
print("Moving on to do computation....\n")

label = dtrain.get_label()
#disable scaling
ratio = float(np.sum(label == 0)) / np.sum(label == 1)
params = {
        "verbosity": 0,
        "booster": "gbtree",
        "objective": "binary:logistic",
        "scale_pos_weight" : ratio,
        "max_delta_step": 1,
        "tree_method": "gpu_hist",
        "eval_metrics": "logloss",
        "learning_rate": 0.09813841335627332,
        "num_boost_round": 281.7702180112571,
        "max-depth": 2, "gamma": 3.8914577758283693e-05,
        "subsample": 0.7341873457611429,
        "reg_alpha": 0.22771113527284387,
        "reg_lambda": 1.6884029915870075e-07,
        "colsample_bytree": 0.850855026540212,
        "min_child_weight": 3,
        "n_estimators": 473
        }

#Load param file
#with open ("old_data_params.txt", 'r') as file:
#    params=json.load(file)

#Define groups of regions
groups = train_df['Region']

#Define splits
gfk = GroupKFold(n_splits=5)
best_iterations = []

#Define a loop
for train_idx, valid_idx in gfk.split(X, y, groups):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    dtrain_fold = xgb.DMatrix(X_train, label=y_train)
    dvalid_fold = xgb.DMatrix(X_valid, label=y_valid)

    bst = xgb.train(
    params,
    dtrain_fold,
    num_boost_round=1000,
    evals=[(dvalid_fold, "validation")],
    early_stopping_rounds=150,
    verbose_eval=False
)
    best_iterations.append(bst.best_iteration)

avg_best_iteration = int(np.mean(best_iterations))

# Retrain on full training set with optimal num_boost_round
bst_final = xgb.train(
    params,
    dtrain,
    num_boost_round=avg_best_iteration,
    evals=[(dtrain, 'train'), (dtest, 'test_2022_2023')],
    early_stopping_rounds=150
)

print("Finished Training...")

# Check accuracy of the model
print("Checking the accuracy of the model...\n")
preds = bst_final.predict(dtest)
labels = dtest.get_label()
error = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
print("error=%f" % error)
print("Model Accuracy: ", 1-error)

# Save the error to a text file
with open("m2_out_of_sample.txt", "w") as f:
    f.write("error=%f\n" % error)


#Save the model
bst.save_model('model2xgb_out_of_sample.json')


