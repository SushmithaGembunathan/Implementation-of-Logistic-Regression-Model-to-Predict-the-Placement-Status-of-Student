# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sushmitha Gembunathan
RegisterNumber:  212224040342
*/
```
```
# 1. Upload the file to Colab
from google.colab import files
uploaded = files.upload()   # Choose Placement_Data.csv from your system

import io
import pandas as pd

data = pd.read_csv(io.BytesIO(uploaded['Placement_Data.csv']))
print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nMissing values:\n", data1.isnull().sum())
print("Duplicates:", data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in ["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation","status"]:
    data1[col] = le.fit_transform(data1[col])

x = data1.iloc[:, :-1]
y = data1["status"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nPredictions:", y_pred)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


sample = [[1,80,1,90,1,1,90,1,0,85,1,85]]  
print("\nSample Prediction:", lr.predict(sample))


```
## Output:

<img width="889" height="1126" alt="image" src="https://github.com/user-attachments/assets/f3a0a2cd-76b4-4f7c-aff9-658e611e26a6" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
