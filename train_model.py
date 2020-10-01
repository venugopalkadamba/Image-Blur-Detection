# importing necessary libraries
import numpy as np
import pandas as pd
import pickle
from scipy.stats import randint

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import warnings
warnings.filterwarnings('ignore')

np.random.seed(24)

# fits the data into model and generates the score on test data
def generate_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    confusion_mat = confusion_matrix(y_test, predictions)
    f1_scre = f1_score(y_test, predictions)
    return accuracy, confusion_mat, f1_scre

models = []
model_names = []
f1_scores = []

train_data = pd.read_csv("train.csv")

print(" About Data ".center(100,'*'))

print("Columns: ", list(train_data.columns))
print("Shape: ", train_data.shape)



# Sepearting features and target variable
X = train_data.iloc[:,:2]
y = train_data.iloc[:, 2]



# Splitting the data for training and testing
print()
print(" Splitting the data into Train and Test ".center(100, "*"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)

print("Training Data: ", X_train.shape, y_train.shape)
print("Test Data: ", X_test.shape, y_test.shape)




# Training Support Vector Machine
print()
print(" Training Support Vector Machine ".center(100,"*"))
svm_model = SVC(C = 10, class_weight = "balanced", gamma = 0.1)
accuracy, confusion_mat, f1_scre = generate_score(svm_model, X_train, y_train, X_test, y_test)
print(f"Accuracy: {accuracy*100}%")
print(f"F1 Score: {f1_scre}")
print("Confusion Matrix: ", confusion_mat)
models.append(svm_model)
f1_scores.append(f1_scre)
model_names.append("SVM")




# Training Random Forest Classifier with Hyperparameter Optimization
print()
print(" Training Random Forest Classifier with Hyperparameter Optimization ".center(100,"*"))
param_dist={'max_depth':[3,5,7,10,None],
              'n_estimators':[10,50,100,150,200,250,300,400,500],
              'max_features':randint(1,2),
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,2),
              }
rf_model = RandomForestClassifier(class_weight='balanced', random_state = 42)
search_clfr = RandomizedSearchCV(rf_model, param_distributions = param_dist, n_jobs=-1, n_iter = 10, cv = 3, verbose =1)
search_clfr.fit(X_train, y_train)
rf_model = search_clfr.best_estimator_
accuracy, confusion_mat, f1_scre = generate_score(rf_model, X_train, y_train, X_test, y_test)
print(f"Accuracy: {accuracy*100}%")
print(f"F1 Score: {f1_scre}")
print("Confusion Matrix: ", confusion_mat)
models.append(rf_model)
f1_scores.append(f1_scre)
model_names.append("RandomForest")



# Training XGBoost Classifier with Hyperparameter Optimization
print()
print(" Training XGBoost Classifier with Hyperparameter Optimization ".center(100,"*"))
xgb_params = {
        'n_estimators':[100,200,300,400,500],
        'learning_rate':[0.01,0.005,0.1,0.002,1],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb_model = XGBClassifier(random_state = 42)
search_clfr = RandomizedSearchCV(xgb_model, param_distributions = xgb_params, n_jobs=-1, n_iter = 10, cv = 3, verbose =1)
search_clfr.fit(X_train, y_train)
xgb_model = search_clfr.best_estimator_
accuracy, confusion_mat, f1_scre = generate_score(xgb_model, X_train, y_train, X_test, y_test)
print(f"Accuracy: {accuracy*100}%")
print(f"F1 Score: {f1_scre}")
print("Confusion Matrix: ", confusion_mat)
models.append(xgb_model)
f1_scores.append(f1_scre)
model_names.append("XGBoost")




# Training Voting Classifier with Hyperparameter Optimization
print()
print(" Training Voting Classifier ".center(100,"*"))
voting_model = VotingClassifier(estimators = [('svc', svm_model),('rf', rf_model),('xgb', xgb_model)], voting = "hard")
accuracy, confusion_mat, f1_scre = generate_score(voting_model, X_train, y_train, X_test, y_test)
print(f"Accuracy: {accuracy*100}%")
print(f"F1 Score: {f1_scre}")
print("Confusion Matrix: ", confusion_mat)
models.append(voting_model)
f1_scores.append(f1_scre)
model_names.append("VotingClassifier")




# Fitting model on whole data and saving them
print()
print("Fitting the voting classifier on whole dataset...")
voting_model.fit(X, y)
pickle.dump(voting_model, open("voting_model.pkl",'wb'))
print("Process done...")
print()
print("Fitting the best classifier among Support Vector Machine, Random Forest and XGBoost on whole dataset...")
model = models[f1_scores.index(max(f1_scores))]
name = model_names[f1_scores.index(max(f1_scores))]
model.fit(X, y)
pickle.dump(voting_model, open(name+".pkl",'wb'))
print("Process done...")
print("Models trained and saved in current working directory.")