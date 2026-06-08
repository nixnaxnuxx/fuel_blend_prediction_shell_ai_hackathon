# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 14:08:08 2025

@author: User
"""

# predict_tabular_models.py

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# === Load Test Data ===
df_test = pd.read_csv('../dataset/test.csv')
df_train = pd.read_csv('../dataset/train.csv')
target_columns = df_train.columns[55:65]
df_test.columns = ['ID'] + [f"f_{i}" for i in range(55)]

X_test = df_test.iloc[:, 1:]

# === Settings ===
VERSION = 'custom_tabular_v1'
MODEL_DIR = f'../models/{VERSION}/'
SAVE_DIR = f'../predicted/{VERSION}/'
os.makedirs(SAVE_DIR, exist_ok=True)

# === Manual Feature Map (Same as training)
target_features = {
    0: ['f_4', 'f_1', 'f_3', 'f_7', 'f_8', 'f_5', 'f_9', 'f_0', 'f_6', 'f_2'],
    1: ['f_4', 'f_1', 'f_13', 'f_3', 'f_2', 'f_14', 'f_12', 'f_0', 'f_10', 'f_11'],
    2: ['f_1', 'f_3', 'f_2', 'f_36', 'f_35', 'f_4', 'f_37', 'f_0', 'f_41', 'f_38'],
    3: ['f_4', 'f_1', 'f_3', 'f_22', 'f_23', 'f_20', 'f_0', 'f_2', 'f_21', 'f_24'],
    4: ['f_1', 'f_26', 'f_25', 'f_0', 'f_27', 'f_2', 'f_22', 'f_32', 'f_12'],
    5: ['f_4', 'f_1', 'f_33', 'f_32', 'f_30', 'f_2', 'f_31', 'f_34', 'f_3', 'f_0'],
    6: ['f_1', 'f_3', 'f_2', 'f_36', 'f_35', 'f_37', 'f_4', 'f_0', 'f_38', 'f_41'],
    7: ['f_1', 'f_4', 'f_43', 'f_42', 'f_3', 'f_40', 'f_2', 'f_0', 'f_46', 'f_39'],
    8: ['f_4', 'f_48', 'f_44', 'f_1', 'f_3', 'f_47', 'f_0', 'f_2', 'f_32', 'f_49'],
    9: ['f_3', 'f_4', 'f_1', 'f_52', 'f_0', 'f_50', 'f_2', 'f_53', 'f_51'],
}

# === Predict All Targets ===
predictions = {}
for i in range(10):
    print(f"Predicting target_{i}")
    model_path = f"{MODEL_DIR}/target_{i}.pkl"
    model = joblib.load(model_path)
    poly = joblib.load(f'{MODEL_DIR}/poly_transformer_{i}.pkl')

    features = target_features[i]
    X_test_sub_raw = X_test[features]
    X_test_sub = poly.transform(X_test_sub_raw)

    y_pred = model.predict(X_test_sub)
    predictions[f"target_{i}"] = y_pred

# === Combine All Predictions into One DataFrame ===
df_predictions = pd.DataFrame(predictions)
df_predictions.columns = target_columns
df_predictions.insert(0, 'ID', np.arange(1, 501))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_predictions.to_csv(f"{SAVE_DIR}/final_predictions_{timestamp}.csv", index=False)

print("\n\033[1;32mFinal predictions saved to final_predictions.csv\033[0m")