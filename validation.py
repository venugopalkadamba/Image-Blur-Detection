import pandas as pd
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import warnings
warnings.filterwarnings('ignore')

validation_data = pd.read_csv("validation.csv")

voting_model = pickle.load(open("voting_model.pkl", "rb"))
model = pickle.load(open("XGBoost.pkl","rb"))

X = validation_data.iloc[:, :2]
y = validation_data.iloc[:, 2]

print("Loading validation data and models successfull...")

voting_predictions = voting_model.predict(X)
xgb_predictions = model.predict(X)

print()
print(" Voting Model Score on Validation Data ".center(100,"*"))
print("Accuracy Score: ", accuracy_score(y, voting_predictions))
print("F1 Score: ", f1_score(y, voting_predictions))
print("Confusion Matrix: ", confusion_matrix(y, voting_predictions))

print()
print(" XGBoost Model Score on Validation Data ".center(100,"*"))
print("Accuracy Score: ", accuracy_score(y, xgb_predictions))
print("F1 Score: ", f1_score(y, xgb_predictions))
print("Confusion Matrix: ", confusion_matrix(y, xgb_predictions))