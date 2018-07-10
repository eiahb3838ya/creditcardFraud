import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek 
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

random_seed = 13

data = pd.read_csv("./creditcard.csv")
X = data.loc[:, data.columns != 'Class'].values
y = data.loc[:, data.columns == 'Class'].values.reshape((len(X), ))

# train-test split
split = StratifiedShuffleSplit(test_size=0.1, random_state=random_seed)
for train_index, test_index in split.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# initilize sampling models
smt = SMOTETomek(random_state=random_seed)
X_res, y_res = smt.fit_sample(X_train, y_train)

# Ridge Logstic Regression
ridge = RidgeClassifierCV(alphas=np.geomspace(1e-5, 10, 100), 
                          cv=10, 
                          class_weight=None)
ridge.fit(X_res, y_res)
print('\n', classification_report_imbalanced(y_test, ridge.predict(X_test), 
                                             target_names=['normal', 'fraud']))

# SVM
svm = LinearSVC(dual=False, verbose=1, 
                random_state=random_seed, 
                max_iter=int(1e6), 
                class_weight=None)
svm.fit(X_res, y_res)
print('\n', classification_report_imbalanced(y_test, svm.predict(X_test), 
                                             target_names=['normal', 'fraud']))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                            random_state=random_seed, verbose=1, 
                            class_weight=None)
rf.fit(X_res, y_res)
print('\n', classification_report_imbalanced(y_test, rf.predict(X_test), 
                                             target_names=['normal', 'fraud']))

# KNN
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_res, y_res)
print('\n', classification_report_imbalanced(y_test, knn.predict(X_test), 
                                             target_names=['normal', 'fraud']))