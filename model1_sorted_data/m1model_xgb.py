import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost import cv
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter 
import os
import sys
#import optuna


path=os.getcwd()
#print(path)


#Load Train and Test Excel files
print("LOADING DATA WAIT.......\n")

train_df = pd.read_csv("m1train_poultry_filled.csv")
test_df = pd.read_csv("m1test_poultry_filled.csv")
X_train = train_df.iloc[:,3:] #Training
y_train = train_df.iloc[:,2:3]

print("Training data\n", X_train)
print("Training labels\n", y_train)
#uncomment to print data
#sys.exit()


X_test = test_df.iloc[:,3:]
X_test_raw = test_df.iloc[:,3:]
y_test = test_df.iloc[:,2:3]
y_test_raw = test_df.iloc[:,2:3]
#print("Testing data\n", X_test)
#print("Testing labels\n", y_test)
#print("Raw Testing Testing data\n", y_test)
#print("Raw Testing labels\n", y_test_raw)
#sys.exit()
#Validation_set
valid= train_df[train_df['Year'] == 2021]
dvalid_data = valid.iloc[:,3:]
dvalid_label = valid.iloc[:,2:3]
dvalid = xgb.DMatrix(dvalid_data, label=dvalid_label, missing=-999.0) #validation set
#print("Validation data\n", dvalid_data)
#print("Validation label\n", dvalid_label)
#sys.exit()
#First we have to convert these values to XgbDMatrix
#Training set
dtrain = xgb.DMatrix(X_train, label=y_train, missing=-999.0) # = xgb.DMatrix(y_train)
#Validation set
dtest = xgb.DMatrix(X_test, label=y_test, missing=-999.0) # = xgb.DMatrix(y_test)
#Test set
X_test = xgb.DMatrix(X_test)
y_test = xgb.DMatrix(y_test)
#Get the labels and scale the data
label = dtrain.get_label()
ratio = float(np.sum(label == 0)) / np.sum(label == 1)
base_params = {
	'verbosity': 0,
	'booster': 'gbtree',
	'objective': 'binary:logistic',
        'scale_pos_weight' : ratio,
        'tree_method': 'gpu_hist',
        'eval_metrics': 'logloss',
        'random_state': 42
}
params = {
          'learning_rate': 0.09963558437961703, 
          'num_boost_round': 191.58714451312179, 
          'max-depth': 4, 
          'gamma': 6.140489891496016e-05, 
          'subsample': 0.7329970919203579, 
          'reg_alpha': 1.5492015574694834e-05, 
          'reg_lambda': 0.002889192981174229, 
          'colsample_bytree': 0.7341652208091972, 
          'min_child_weight': 0, 
          'n_estimators': 443
          }

params.update(base_params)
print(params)
#sys.exit()
xgb_cv = cv(dtrain=dtrain,
            params = params,
           nfold=5,
           num_boost_round=5000,
           early_stopping_rounds=150,
           metrics='logloss',
           as_pandas=True,
           seed=0,
           verbose_eval=1
           )
print("------------------------------------------------------------#\n")
print("-------------------Finished Cross-Validation-------------------\n")
print("------------------------------------------------------------#\n")

#Save the plot of the cross-validation graph
#get train_log_loss-mean/std

tmean = xgb_cv['train-logloss-mean']
testmean = xgb_cv['test-logloss-mean']
plt.plot(tmean, label = "Train")
plt.plot(testmean, label="Validation")
plt.title("Cross-Validation curve for logloss mean")
plt.xlabel("n_rounds")
plt.ylabel("logloss")
plt.legend(loc="upper right")
plt.savefig("xgb_cv_plot.png", dpi=300)

#Rerun training with the best hyperparameter combination
print("Rerunning using the best trial params...")
print("params = {}".format(params))
bst = xgb.train(params, dtrain,
        num_boost_round=10000,
	evals=[(dtrain, 'train'), (dvalid, 'valid_2021'), (dtest, 'test_2021')],
        #n_tree_limit = bst.best_iteration,
	early_stopping_rounds=150)
print("Finished Training...")

#Check accuracy of the model
print("Checking the acurracy of the model...\n")
preds = bst.predict(dtest)
labels = dtest.get_label()
print(
    "error=%f"
    % (
        sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])
        / float(len(preds))
    )
)


#print(mypredict)
#Save config and the model
config = bst.save_config()
print(config)
#Save the hyperparameters
with open('model1_params.json', 'w') as f:
    json.dump(params, f)
print("succesfully saved file\n")
bst.save_model('model1xgb_poltry.json')


