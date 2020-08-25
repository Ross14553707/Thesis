import time
start_time = time.time()

# The necessary packages are loaded in
import sklearn
import numpy as np
import pandas as pd

from sklearn import ensemble

# The values from the datasets are then loaded in, with "data" being the instances and 
# "datay" being the numerical representation of the class
data = pd.read_csv('appdatamix3norm.csv')
datay = pd.read_csv('appdatamix3y.csv')

print("Datasets loaded")
print("--- %s seconds ---" % round((time.time() - start_time), 2))

# Only certain attributes are loaded in to the model, chosen using the LSVC included 
# in the KNN code
X = data.values[:, [10, 11, 14, 18, 40, 41, 46, 57, 58, 68, 70, 72, 73, 75]]
y = datay.values[:, 1]

# The model is defined using 100 classifiers
clf = ensemble.RandomForestClassifier(n_estimators=100)

from sklearn.model_selection import cross_val_score, cross_val_predict

# Evaluate model with cross-validation of 5 folds
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')

# Uncommenting this block of code will allow the confusion matrix to be printed
#from sklearn.metrics import confusion_matrix
#y_pred =  cross_val_predict(clf, X, y, cv=5)
#conf_mat = confusion_matrix(y, y_pred)
#print(conf_mat)

# Print each cv score (weighted F-score) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))
print("--- %s seconds ---" % round((time.time() - start_time), 2)) 
   
