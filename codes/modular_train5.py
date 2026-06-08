import pandas as pd
import numpy as np
import joblib
import os
import json
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures

# === SETTINGS ===
VERSION = 'custom_tabular_v1'
MODEL_DIR = f'../models/{VERSION}/'
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv('../dataset/train.csv').astype(float)
df.columns = [f"f_{i}" if i < 56 else f"t_{i-56}" for i in range(df.shape[1])]
X_all = df.iloc[:, :55]
y_all = df.iloc[:, 55:65]

# === Custom Config ===
target_configs = {
    0: {'features': ['f_4', 'f_1', 'f_3', 'f_7', 'f_8', 'f_5', 'f_9', 'f_0', 'f_6', 'f_2'], 'model': 'stack', 'meta': 'xgb'},
    1: {'features': ['f_4', 'f_1', 'f_13', 'f_3', 'f_2', 'f_14', 'f_12', 'f_0', 'f_10', 'f_11'], 'model': 'xgb'},
    2: {'features': ['f_1', 'f_3', 'f_2', 'f_36', 'f_35', 'f_4', 'f_37', 'f_0', 'f_41', 'f_38'], 'model': 'stack', 'meta': 'ridge'},
    3: {'features': ['f_4', 'f_1', 'f_3', 'f_22', 'f_23', 'f_20', 'f_0', 'f_2', 'f_21', 'f_24'], 'model': 'stack', 'meta':'ridge'},
    4: {'features': ['f_1', 'f_26', 'f_25', 'f_0', 'f_27', 'f_2', 'f_22', 'f_32', 'f_12'], 'model': 'xgb'},
    5: {'features': ['f_4', 'f_1', 'f_33', 'f_32', 'f_30', 'f_2', 'f_31', 'f_34', 'f_3', 'f_0'], 'model': 'stack', 'meta': 'ridge'},
    6: {'features': ['f_1', 'f_3', 'f_2', 'f_36', 'f_35', 'f_37', 'f_4', 'f_0', 'f_38', 'f_41'], 'model': 'lgb'},
    7: {'features': ['f_1', 'f_4', 'f_43', 'f_42', 'f_3', 'f_40', 'f_2', 'f_0', 'f_46', 'f_39'], 'model': 'xgb'},
    8: {'features': ['f_4', 'f_48', 'f_44', 'f_1', 'f_3', 'f_47', 'f_0', 'f_2', 'f_32', 'f_49'], 'model': 'xgb'},
    9: {'features': ['f_3', 'f_4', 'f_1', 'f_52', 'f_0', 'f_50', 'f_2', 'f_53', 'f_51'], 'model': 'stack', 'meta' : 'ridge'},
}

# === Objective Function with K-Fold CV ===
def objective(trial, model_type, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mape_scores = []

    # Create polynomial features inside CV folds to avoid data leakage
    poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply polynomial features
        X_train_poly = poly.fit_transform(X_train_fold)
        X_val_poly = poly.transform(X_val_fold)
        
        if model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'eval_metric': 'rmse',
                'use_label_encoder': False,
            }
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train_poly, y_train_fold,
                eval_set=[(X_val_poly, y_val_fold)]
            )
            
        elif model_type == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                'random_state': 42,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_poly, y_train_fold,
                eval_set=[(X_val_poly, y_val_fold)]
            )
            
        elif model_type == 'ridge':
            alphas = trial.suggest_categorical('alphas', [[0.001, 0.01, 0.1, 1, 10, 100]])
            model = RidgeCV(alphas=alphas)
            model.fit(X_train_poly, y_train_fold)

        y_pred = model.predict(X_val_poly)
        fold_mape = mean_absolute_percentage_error(y_val_fold, y_pred)
        mape_scores.append(fold_mape)

    return np.mean(mape_scores)

# === Train Loop ===
for i, cfg in target_configs.items():
    print(f"\n🎯 Training target_{i} with {cfg['model']}...")

    features = cfg['features']
    X_sub = X_all[features]
    y_sub = y_all.iloc[:, i]

    # Create polynomial transformer to save for inference
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_sub)

    model_type = cfg['model'] if cfg['model'] != 'stack' else cfg.get('meta', 'ridge')
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, model_type, X_sub, y_sub), n_trials=50)
    
    best_params = study.best_params

    # Train final model on full data with polynomial features
    if cfg['model'] == 'xgb':
        model = xgb.XGBRegressor(**best_params, random_state=42)
        model.fit(X_poly, y_sub)
    elif cfg['model'] == 'lgb':
        model = lgb.LGBMRegressor(**best_params, random_state=42)
        model.fit(X_poly, y_sub)
    elif cfg['model'] == 'stack':
        base_models = [
            ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
            ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42)),
        ]
        meta_model_type = cfg.get('meta', 'ridge')
        meta_model = RidgeCV(**best_params) if meta_model_type == 'ridge' else xgb.XGBRegressor(**best_params, random_state=42)
        model = StackingRegressor(estimators=base_models, final_estimator=meta_model, passthrough=True)
        model.fit(X_poly, y_sub)
    
    # Save model and polynomial transformer
    joblib.dump(model, f'{MODEL_DIR}/target_{i}.pkl')
    joblib.dump(poly, f'{MODEL_DIR}/poly_transformer_{i}.pkl')

    with open(f"{MODEL_DIR}/target_{i}_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)

    print(f"✅ Saved model and poly transformer for target_{i}")

print("\n\033[1;32mAll models saved with cross-validation tuning.\033[0m")
