# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries.
2. Load the dataset using pd.read_csv().
3. Display data types, basic statistics, and class distributions.
4. Visualize class distributions with a bar plot.
5. Scale feature columns using MinMaxScaler.
6. Encode target labels with LabelEncoder.
7. Split data into training and testing sets with train_test_split().
8. Train LogisticRegression with specified hyperparameters and evaluate the model using metrics and a confusion matrix plot. 

## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: Kesav Deepak Sridharan
RegisterNumber: 212223230104
*/
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
food_df = pd.read_csv(dataset_url)

food_df.dtypes
```
![image](https://github.com/user-attachments/assets/39aaa830-2922-47d8-adbb-9de9bfbdcc0a)
```
food_df.head(10)
```
![image](https://github.com/user-attachments/assets/db206f08-c8c1-4a26-83b7-ce7d14a3d630)
```
feature_cols = list(food_df.iloc[:, :-1].columns)
feature_cols
```
![image](https://github.com/user-attachments/assets/7dadebce-e4b0-4499-a30c-b134f31eb915)
```
food_df.iloc[:, :-1].describe()
```
![image](https://github.com/user-attachments/assets/b1e6677f-c5e8-419d-87f4-38450eb4d645)
```
food_df.iloc[:, -1:].value_counts(normalize=True)
```
![image](https://github.com/user-attachments/assets/1a6627e5-e3d8-41f8-ae45-5dce1b3645b9)

```
food_df.iloc[:, -1:].value_counts().plot.bar(color=['yellow', 'red', 'green'])
```
![image](https://github.com/user-attachments/assets/15d4a6ab-ca18-409d-9882-571cdb7245e3)

```
X_raw = food_df.iloc[:, :-1]
y_raw = food_df.iloc[:, -1:]
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
print(f"The range of feature inputs are within {X.min()} to {X.max()}")
```
![image](https://github.com/user-attachments/assets/0dd8afcc-b3df-465d-a4d9-6b44f9828f99)
```
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())
np.unique(y, return_counts=True)
```
![image](https://github.com/user-attachments/assets/1dccee61-4e89-48a1-8ac2-42b36f1b9801)
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)
print(f"Training dataset shape, X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing dataset shape, X_test: {X_test.shape}, y_test: {y_test.shape}")
```
![Screenshot 2024-11-11 015022](https://github.com/user-attachments/assets/e9bd48eb-1d5c-42ae-a485-eb9aba05bb91)

![image](https://github.com/user-attachments/assets/9b12d952-cad6-425c-8928-ec48d6be41df)
```
penalty= 'elasticnet'
multi_class = 'multinomial'
solver = 'saga'
max_iter = 1000
l1_ratio = 0.5
en_model = LogisticRegression(random_state=rs, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter, l1_ratio=l1_ratio)
en_model.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/4a442598-53ed-4c34-a1b0-aedc8f7e2daf)

```
en_preds = en_model.predict(X_test)
en_metrics = evaluate_metrics(y_test, en_preds)
print(en_metrics)
```
![image](https://github.com/user-attachments/assets/54db9f2f-c0fb-435c-bd3e-4abf4a7c20a8)
```
cf_matrix = confusion_matrix(y_test, en_preds, normalize='true')
sns.set_context('talk')
disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=en_model.classes_)
disp.plot()
plt.show()
```
![image](https://github.com/user-attachments/assets/a096fc2b-2938-4624-a460-0a9031535b11)




## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
