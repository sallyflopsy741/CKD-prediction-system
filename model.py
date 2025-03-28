"""import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import sklearn
import logging
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics._classification import classification_report
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from collections import Counter

df = pd.read_csv("Final_pre_processing_data.csv")

print(df.head())
# Feature selection
X = df[["age","bp","sg","al","su","rbc","pc","pcc","ba","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm","cad","appet","pe","ane"]]
y = df[["classification"]]

# Label Encoding for categorical features
le = LabelEncoder()
categorical_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
for col in categorical_cols:
    X.loc[:, col] = le.fit_transform(X[col])
    
# Fix label encoding for y (Ensure correct mapping)
y = le.fit_transform(y)
print("Label Encoding Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

## Use MinMaxScaler instead of StandardScaler to handle categorical-like numeric features
scaler = MinMaxScaler()
num_cols = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
X.loc[:, num_cols] = scaler.fit_transform(X[num_cols])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50, stratify=y)

# Fix shape issue
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Handle class imbalance
#num_ckd = np.sum(y_train == 1)
#num_not_ckd = np.sum(y_train == 0)
#scale_pos_weight = (num_not_ckd / num_ckd)*1.5

# Handle class imbalance properly
counter = Counter(y_train)
scale_pos_weight = counter[0] / counter[1]

# Train XGBoost Model
classifier = XGBClassifier(scale_pos_weight=scale_pos_weight, max_depth=5, learning_rate=0.1, n_estimators=200)
classifier.fit(X_train, y_train)
        


# Predictions using probability-based classification
y_probs = classifier.predict_proba(X_test)[:, 1]  # Get probability of the positive class
threshold = 0.4  # Adjusted threshold to improve recall for positive cases
y_pred = (y_probs > threshold).astype(int)

# Performance Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy:', round(accuracy_score(y_test, y_pred), 2))



#save the model
pickle.dump(classifier,open('model.pkl','wb'))"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE  # For balancing dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Load dataset
df = pd.read_csv("Final_pre_processing_data.csv")

# Drop unnecessary columns
df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors='ignore')

# Encode categorical features
le = LabelEncoder()
categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

y = df["classification"]
X = df.drop(columns=["classification"])

# Normalize skewed numerical features
skewed_features = ["bgr", "bu", "sc", "sod"]
X[skewed_features] = np.log1p(X[skewed_features])

# Standardize numerical features
scaler = StandardScaler()
numeric_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=50)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train XGBoost classifier
classifier = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=200)
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Performance evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

# Save the model
pickle.dump(classifier, open('model.pkl', 'wb'))

    


