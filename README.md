# Import libraries
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from sklearn.datasets import load_iris # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Map target labels to species names
data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display dataset overview
print("Dataset Overview:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nDataset Description:")
print(data.describe())

# Encode target labels for model training
data['species_encoded'] = data['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Split features and target
X = data.iloc[:, :-2]  # All features
y = data['species_encoded']  # Encoded target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Visualize feature distributions by species
sns.pairplot(data, hue='species', diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.iloc[:, :-2].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predictions
y_pred_knn = knn_model.predict(X_test)

# Evaluation
print("KNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Logistic Regression Confusion Matrix
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")

# KNN Confusion Matrix
plot_confusion_matrix(y_test, y_pred_knn, "KNN")

# Save cleaned dataset
data.to_csv('processed_iris.csv', index=False)

# Save trained models
import joblib # type: ignore
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')





