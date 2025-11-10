# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:40:08 2025

@author: robbi
"""

import os
import pandas as pd
import numpy as np

if os.getcwd() == 'C:\\Users\\robbi\\ML projects\\wm_sales':
    os.chdir('walmartforecasting')

features = pd.read_csv('features.csv.zip')
test = pd.read_csv('test.csv.zip')
train = pd.read_csv('train.csv.zip')

merged_train = train.merge(features, on=["Store","Date"], how="left")

for c in merged_train.columns:
    if merged_train[c].isna().any():
        print(c)
    if c.startswith("MarkDown"):
        merged_train[c].fillna(0, inplace=True)

# Sorting Values before lagging values to ensure correct chronologics.
merged_train.sort_values(['Store','Dept','Date'])

# Lagging process
merged_train['lag_1'] = merged_train.groupby(['Store','Date'])['Weekly_Sales'].shift(1)
