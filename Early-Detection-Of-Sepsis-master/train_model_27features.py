#!/usr/bin/env python
# Fast model training with 27 features only

import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
dataset = pd.read_csv("sepsis.csv")

print("Selecting 27 features...")
# Use only these 27 features
feature_cols = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp',
    'BaseExcess', 'HCO3', 'FiO2', 'PaCO2', 'SaO2', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Bilirubin_total', 'Hgb', 'WBC', 'Fibrinogen', 'Platelets',
    'Age', 'Gender', 'HospAdmTime', 'ICULOS'
]

# Filter dataset to only include these columns
dataset = dataset[feature_cols + ['SepsisLabel']]

print("Balancing classes...")
df_majority = dataset[dataset.SepsisLabel==0]
df_minority = dataset[dataset.SepsisLabel==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=123)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print("Preparing features and labels...")
X = df_upsampled[feature_cols].values
Y = df_upsampled['SepsisLabel'].values

labelencoder_Y = preprocessing.LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("\nTraining MLP model (this may take a few minutes)...")

model = MLPClassifier(
    activation='tanh',
    solver='lbfgs',
    early_stopping=False,
    hidden_layer_sizes=(27, 15, 10, 5, 2),
    random_state=1,
    batch_size='auto',
    max_iter=10000,
    learning_rate_init=1e-5,
    tol=1e-4,
    verbose=1
)

model.fit(X_train, Y_train)

print("\nSaving model...")
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved successfully to model.pkl")

# Quick test
from sklearn.metrics import accuracy_score
train_predictions = model.predict(X_test)
acc = accuracy_score(Y_test, train_predictions)
print(f"Model Accuracy on test set: {acc:.4%}")
print(f"Number of features: {len(feature_cols)}")
