import pandas as pd
import numpy as np
import sklearn.svm as svm
import pickle
import math

# clf = pickle.load('lin_clf.joblib')

validate_queries = pd.read_csv('validate_queries.csv')
validate_queries = validate_queries.values
sub = pd.read_csv('linear_submission.csv')
sub = sub.values

num = 0

for i in range(len(validate_queries)):
    num += (int(sub[i][1]) - validate_queries[i][3]) ** 2

print(math.sqrt(float(num) / len(sub)))





