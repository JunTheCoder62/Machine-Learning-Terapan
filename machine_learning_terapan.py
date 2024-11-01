# -*- coding: utf-8 -*-
"""Machine_Learning_Terapan.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v8fnXQTUlA3_LraxZCIkdY2RWRTAIMoj

# Import
"""

import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import phik

# Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Pipeline
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

"""# Data Loading"""

data = 'https://raw.githubusercontent.com/JunTheCoder62/Feed/refs/heads/main/stroke_prediction_dataset.csv'
df = pd.read_csv(data)
df

"""# EDA"""

df = df.drop(['Patient ID', 'Patient Name'], axis = 1)
df.info()

"""## Handilng Missing Value"""

df.isna().sum()

"""terdapat 2500 missing value"""

df = df.dropna(axis=1)

df.isnull().sum()

"""## Handling Duplicate"""

df.duplicated().sum()

"""tidak ada duplikat"""

df.describe(include='object')

"""## Convert Type"""

df.info()

# df['Work Type'] = df['Work Type'].astype('category')
# df['Gender'] = df['Gender'].astype('category')
# df['Smoking Status'] = df['Gender'].astype('category')
# df['Residence Type'] = df['Residence Type'].astype('category')
# df['Alcohol Intake'] = df['Alcohol Intake'].astype('category')
# df['Physical Activity'] = df['Physical Activity'].astype('category')
# df['Family History of Stroke'] = df['Family History of Stroke'].astype('category')
# df['Dietary Habits'] = df['Dietary Habits'].astype('category')
# df['Blood Pressure Levels'] = df['Blood Pressure Levels'].astype('category')
# df['Cholesterol Levels'] = df['Cholesterol Levels'].astype('category')
# df['Diagnosis'] = df['Diagnosis'].astype('category')

df_category = ['Work Type', 'Gender', 'Smoking Status', 'Residence Type',
    'Alcohol Intake', 'Physical Activity', 'Family History of Stroke',
    'Dietary Habits', 'Blood Pressure Levels', 'Cholesterol Levels', 'Diagnosis']
df[df_category] = df[df_category].apply(lambda x: x.astype('category'))
df = df.drop('Marital Status', axis = 1)
df.info()

"""# Business Understanding

## Univariate Analysis
"""

df.info()

df['Diagnosis'].value_counts(normalize=True)

"""data yang ada seimbang. tidak perlu balancing data"""

df[df['Age'] == df['Age'].min()]

df['Gender'].value_counts()

df['Gender'].value_counts().plot(kind='bar')
plt.xticks()
plt.title('Gender')

"""Male category lebih banyak memiliki Stroke

"""

df['Work Type'].value_counts()

df.groupby('Work Type')['Diagnosis'].value_counts()

# Remove Never Worked
df = df[df['Work Type'] != 'Never Worked']
df['Work Type'].value_counts()

df.groupby('Work Type')['Diagnosis'].value_counts()

df['Smoking Status'].value_counts()

df['Smoking Status'].hist()

df['Age'].hist()

df['Diagnosis'].value_counts()

df['Diagnosis'].value_counts().plot(kind='bar')
plt.xticks()
plt.title('Stroke')

"""## Multivariate Analysis"""

# Distribution between Age and Work Type

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x ='Age', hue='Diagnosis', kde=True)

df_plot1 = df[['Diagnosis', 'Work Type']].groupby('Work Type').value_counts().reset_index()
sns.barplot(data=df_plot1, x='Diagnosis', y='count', hue='Work Type')
plt.title('Stroke and Work Type')

"""## Features Engineering"""

num_features = ['Age', 'Hypertension', 'Heart Disease', 'Stroke History', 'Stress Levels']
cat_features = ['Gender', 'Work Type', 'Residence Type', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Family History of Stroke', 'Dietary Habits', 'Blood Pressure Levels', 'Cholesterol Levels', 'Diagnosis']

features = cat_features[0]

counter = df[features].value_counts()
persen = 100 * df[features].value_counts(normalize = True)
temp = pd.DataFrame({'data': counter, 'persentase': persen.round(1)})
print(temp)

counter.plot(kind='bar', title="Fitur " + features);

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

features = cat_features[1]

counter = df[features].value_counts()
persen = 100 * df[features].value_counts(normalize = True)
temp = pd.DataFrame({'data': counter, 'persentase': persen.round(1)})
print(temp)

counter.plot(kind='bar', title="Fitur " + features);

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

df.isna().sum()

df.shape

df.dropna(inplace=True)

# Correlation Matrix

df_numeric = df.select_dtypes(include=[np.number])
corr = df_numeric.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(data=corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title(" Correlation Matrix Between Numerical Feature", size = 20)
plt.show()

"""# Data Preperation

## Encoder Data
"""

from imblearn.under_sampling import RandomUnderSampler

# Using OnehotEncoder

# ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
# df_encoder = ohe.fit_transform(df[['Gender','Diagnosis', 'Work Type', 'Residence Type', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Family History of Stroke', 'Dietary Habits']])
# df_encoder

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Diagnosis'] = encoder.fit_transform(df['Diagnosis'])
df['Work Type'] = encoder.fit_transform(df['Residence Type'])
df['Smoking Status'] = encoder.fit_transform(df['Smoking Status'])
df['Alcohol Intake'] = encoder.fit_transform(df['Alcohol Intake'])
df['Physical Activity'] = encoder.fit_transform(df['Physical Activity'])
df['Family History of Stroke'] = encoder.fit_transform(df['Family History of Stroke'])
df['Dietary Habits'] = encoder.fit_transform(df['Diagnosis'])
df['Residence Type'] = encoder.fit_transform(df['Residence Type'])
df['Blood Pressure Levels'] = encoder.fit_transform(df['Blood Pressure Levels'])
df['Cholesterol Levels'] = encoder.fit_transform(df['Cholesterol Levels'])
df.head()

# df = pd.get_dummies(df[['Work Type', 'Residence Type', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Family History of Stroke', 'Dietary Habits', 'Diagnosis']], drop_first=True).astype(int)
# df.head()

# df_enc = pd.concat([df, df_encoder], axis=1).drop(columns=['Gender', 'Work Type', 'Residence Type', 'Smoking Status', 'Alcohol Intake', 'Physical Activity', 'Family History of Stroke', 'Dietary Habits', 'Blood Pressure Levels', 'Cholesterol Levels'])
# df_enc.head(7)

# df_enc.info()

corr = df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(data=corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title(" Correlation Matrix Between Numerical Feature", size = 20)
plt.show()

"""Dapat dilihat bahwa fitur Diagnosis dan Dietary Habits dapat digunakan sebagai model."""

df.corr()

df.info()

# corr = df.corr()
# plt.figure(figsize=(10, 6))
# sns.heatmap(data=corr, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title(" Correlation Matrix Between Numerical Feature", size = 20)
# plt.show()

# scalling
scaler = StandardScaler()

# split data test dan train
X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

#  balancing Data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print([len(X_train), len(y_train)])
print([len(X_test), len(y_test)])

y_train.value_counts(normalize=True)

y_test.value_counts(normalize=True)

"""# Modeling"""

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

"""## Random Forest"""

# RandomForest

modem_rm = RandomForestClassifier()
modem_rm.fit(X_train, y_train)
predic_rf = modem_rm.predict(X_test)

print("akurasi:", accuracy_score(y_test, predic_rf))
print(classification_report(y_test, predic_rf))

# Confusion Matrix

rm_cm = confusion_matrix(y_test, predic_rf)
rm_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=rm_cm)
rm_cm.plot()

"""## Decision Tree"""

# Decision Tree

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
predic_dt = model_dt.predict(X_test)

print("akurasi:", accuracy_score(y_test, predic_dt))
print(classification_report(y_test, predic_dt))

# Confusion Matrix

dt_cm = confusion_matrix(y_test, predic_dt)
dt_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=dt_cm)
dt_cm.plot()

"""## KNN model"""

# KNN model

model_knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute', weights='uniform')
model_knn.fit(X_train, y_train)
predic_knn = model_knn.predict(X_test)

print("akurasi:", accuracy_score(y_test, predic_knn))
print(classification_report(y_test, predic_knn))

# Confusion Matrix

knn_cm = confusion_matrix(y_test, predic_knn)
knn_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=knn_cm)
knn_cm.plot()

"""## AdaBoost"""

# AdaBoost

model_ab = AdaBoostClassifier(estimator=LogisticRegression(random_state=42),
                              n_estimators=50,
                              random_state=42)
model_ab.fit(X_train, y_train)
predic_ab = model_ab.predict(X_test)

print("akurasi:", accuracy_score(y_test, predic_ab))
print(classification_report(y_test, predic_ab))

# Confusion Matrix

ab_cm = confusion_matrix(y_test, predic_ab)
ab_cm = metrics.ConfusionMatrixDisplay(confusion_matrix=ab_cm)
ab_cm.plot()

"""## Hyperparameter Tuning"""

model_rf = RandomForestClassifier()

# Tentukan parameter yang akan dicari nilainya dalam hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Lakukan pencarian hyperparameter dengan GridSearchCV
grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Tampilkan parameter terbaik
print("Best Parameters:", grid_search.best_params_)

# Gunakan model dengan parameter terbaik
best_model = grid_search.best_estimator_
predic_rf = best_model.predict(X_test)

# Evaluasi model
print("Akurasi:", accuracy_score(y_test, predic_rf))
print(classification_report(y_test, predic_rf))

# Hitung Mean Squared Error
mse = mean_squared_error(y_test, predic_rf)
print("Mean Squared Error:", mse)