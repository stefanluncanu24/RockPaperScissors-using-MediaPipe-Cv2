import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
dataset = pd.read_csv('rps.txt', header=None)

# Split the dataset into features and labels
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier()
model.fit(X_train, y_train)


# Evaluate the model
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save the model
model.save_model('xgb_model.json')

