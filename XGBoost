import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data
X_train = np.array([[1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

# Train-test split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert the datasets into an optimized data structure called DMatrix
dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
dtest = xgb.DMatrix(X_test_split, label=y_test_split)

# Define the XGBoost parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic'
}

# Number of boosting rounds
num_round = 100

# Train the model
bst = xgb.train(params, dtrain, num_round)

# Make predictions
y_pred_prob = bst.predict(dtest)
y_pred = np.where(y_pred_prob >= 0.5, 1, 0)

# Accuracy
accuracy = accuracy_score(y_test_split, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Example prediction
example = np.array([[1, 1, 1]])  # Pointy ears, round face shape, whiskers present
dexample = xgb.DMatrix(example)
prediction_prob = bst.predict(dexample)
prediction = np.where(prediction_prob >= 0.5, 1, 0)
print(f'Is it a cat? {"Yes" if prediction[0] == 1 else "No"}')

