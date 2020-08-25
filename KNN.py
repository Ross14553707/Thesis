import time
start_time = time.time()

# The necessary packages are loaded in
import sklearn
import numpy as np
import pandas as pd

from sklearn import neighbors

# The values from the datasets are then loaded in, with "data" being the instances and 
# "datay" being the numerical representation of the class
data = pd.read_csv('appdatamix2norm.csv')
datay = pd.read_csv('appdatamix2y.csv')

print("Datasets loaded")
print("--- %s seconds ---" % round((time.time() - start_time), 2))


# Only certain attributes are loaded in to the model, chosen using the LSVC included 
# below
X = data.values[:, [14, 18, 40, 41, 46, 57, 58, 68, 70, 72, 73, 75]]
y = datay.values[:,1]

### Linear Support Vector Classification ###

# This block of code can take in all of the attributes in the dataset if needed and 
# return the attributes that most correlate with the application class
#from sklearn.svm import LinearSVC
#from sklearn.feature_selection import SelectFromModel
#lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(X, y)
#model = SelectFromModel(lsvc, prefit=True)
#X_new = model.transform(X)

#print("New model created")
#print("--- %s seconds ---" % (time.time() - start_time)) 


# Here, the shape of the new dataset is shown as well as the first line,
# so that the column numbers can be figured out
#print(X_new.shape)
#print(X_new[0,:])

######


from sklearn.model_selection import cross_val_score, cross_val_predict


# The model is defined with the value for K set
n_neighbors = 7
knn_cv = neighbors.KNeighborsClassifier(n_neighbors, weights = 'distance', metric = 'manhattan')

# Evaluate model with cross-validation of 5 folds
cv_scores = cross_val_score(knn_cv, X, y, cv=5, scoring='f1_weighted')

# Uncommenting this block of code will allow the confusion matrix to be printed
#from sklearn.metrics import confusion_matrix
#y_pred =  cross_val_predict(knn_cv, X, y, cv=5)
#conf_mat = confusion_matrix(y, y_pred)
#print(conf_mat)

# Print each cv score (weighted F-score) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores))) 
print("--- %s seconds ---" % round((time.time() - start_time), 2))  

