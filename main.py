# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:40:08 2025

@author: robbi
"""

import os
import pandas as pd
import numpy as np

if os.getcwd() == 'C:\\Users\\robbi':
    os.chdir('ML projects\\wm_sales\\walmartforecasting')

features = pd.read_csv('features.csv.zip')
test = pd.read_csv('test.csv.zip')
train = pd.read_csv('train.csv.zip')

merged_train = train.merge(features, on=["Store","Date"], how="left")

for c in merged_train.columns:
    # if merged_train[c].isna().any():
    #     print(c)
    if c.startswith("MarkDown"):
        merged_train[c].fillna(0, inplace=True)

# Sorting Values before lagging values to ensure correct chronologics.
merged_train = merged_train.sort_values(['Store','Dept','Date']).reset_index(drop=True)

# Ensuring date values are datetime and not string
merged_train['Date'] = pd.to_datetime(merged_train['Date'])

# Lagging process
# merged_train['lag_1'] = merged_train.groupby(['Store','Date'])['Weekly_Sales'].shift(1)

# Adding 4 week lag process (for 4 week rolling average)
for k in [1,2,3,4]:
    merged_train[f"lag_{k}"] = (
        merged_train
        .groupby(["Store","Dept"])["Weekly_Sales"]
        .shift(k)
    )
    
# Implementing rolling average
merged_train["roll_mean_4"] = (
    merged_train
    .groupby(["Store","Dept"])["Weekly_Sales"]
    .apply(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())
)

# Creating new merged df without empty lag_1s
merged_model = merged_train.dropna(subset=["lag_1"]).copy()

# Exogenous variables
exog = []
for c in merged_train.columns:
    if c in ["Temperature", "CPI", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "Fuel_Price", "Unemployment", "IsHoliday"]:
        exog.append(c)