import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def find_mse(predicted_values_file):
    pred_values  = pd.read_csv(predicted_values_file)
    print(pred_values.shape)
    stars_pred = pred_values['stars']

    true_values  = pd.read_csv('test_with_gt_cleaned.csv')
    print(true_values.shape)
    stars_true  = true_values['stars']

    mse = mean_squared_error(stars_true, stars_pred)**0.5

    return mse

#find_mse('out_validate_1.csv')