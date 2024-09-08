# train_save_model.py

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
cardio_data = pd.read_csv("HEART_MODEL/CAD/Coronary_Artery_Disease.csv")

# Separate features and target
x = cardio_data.drop(columns="Cath", axis=1)
y = cardio_data["Cath"]

# Feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=2)

# Initialize individual models with tuned parameters
log_reg = LogisticRegression(max_iter=1000, C=0.1, penalty='l2')
svm = SVC(kernel='rbf', C=1.0, probability=True)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5)
knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3)

# Define a VotingClassifier with optimized weights
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', log_reg),
        ('svm', svm),
        ('rf', rf),
        ('knn', knn),
        ('gb', gb)
    ],
    voting='soft',
    weights=[1, 2, 1, 1, 2]  # Adjusted weights based on individual model performance
)

# Train the ensemble model on the entire training set
ensemble_model.fit(x_train, y_train)

# Save the trained model to a pickle file
with open('HEART_MODEL/CAD/cad_prediction.pkl', 'wb') as model_file:
    pickle.dump(ensemble_model, model_file)

print("Model trained and saved successfully.")
