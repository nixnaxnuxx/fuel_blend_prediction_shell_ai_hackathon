#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import optuna
from pycaret.regression import *


# In[18]:


# Load and Prepare Data

path = r'../dataset/'

df_train = pd.read_csv(path + 'train.csv').astype(float) #ensure all values are float type

X = np.array(df_train.iloc[:, :55])
y_all = np.array(df_train.iloc[:, 55:65])

X_train, X_test, y_train, y_test = train_test_split(X, y_all, test_size=0.2, random_state=42)


# In[18]:


# AutoGluon for Automated Feature Selection & Stacking
from autogluon.tabular import TabularPredictor
ag_data = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X.shape[1])])
ag_data['target'] = y_train[:, 0]  # repeat per target
predictor = TabularPredictor(label='target').fit(ag_data)


# In[ ]:


predictor.leaderboard(silent=True)

