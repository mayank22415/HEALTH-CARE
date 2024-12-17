import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Heart Disease Prediction
# Step 1: Load the Dataset
heart_data = pd.read_csv('heart.csv')  # Replace with the path to your dataset

# Step 2: Preprocessing
X_heart = heart_data.drop('target', axis=1)  # Features
y_heart = heart_data['target']  # Target variable

# Standardize the data
scaler = StandardScaler()
X_heart = scaler.fit_transform(X_heart)

# Step 3: Split the data
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42
)

# Step 4: Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_heart, y_train_heart)

# Step 5: Evaluate the model
heart_predictions = log_reg.predict(X_test_heart)
print("Heart Disease Prediction Report:\n")
print(classification_report(y_test_heart, heart_predictions))

# Breast Cancer Diagnosis
# Step 1: Load the Dataset
cancer_data = pd.read_csv('breast_cancer.csv')  # Replace with the path to your dataset

# Step 2: Preprocessing
X_cancer = cancer_data.drop('diagnosis', axis=1)  # Features
y_cancer = cancer_data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)  # Convert diagnosis to binary

# Standardize the data
X_cancer = scaler.fit_transform(X_cancer)

# Step 3: Split the data
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

# Step 4: Train KNN model with cross-validation
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X_train_cancer, y_train_cancer, cv=5)
print(f"KNN Cross-Validation Accuracy: {np.mean(scores):.2f}")

# Train final KNN model
knn.fit(X_train_cancer, y_train_cancer)

# Step 5: Evaluate the model
cancer_predictions = knn.predict(X_test_cancer)
print("\nBreast Cancer Diagnosis Report:\n")
print(classification_report(y_test_cancer, cancer_predictions))

# Final Accuracy Scores
print(f"Heart Disease Prediction Accuracy: {accuracy_score(y_test_heart, heart_predictions):.2f}")
print(f"Breast Cancer Diagnosis Accuracy: {accuracy_score(y_test_cancer, cancer_predictions):.2f}")