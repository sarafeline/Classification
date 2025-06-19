import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data from a CSV file
file_path = 'feline_data.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Convert 'Lion' to 1 and 'Cat' to 0 in the label column
data['Lion_or_Cat'] = data['Lion_or_Cat'].map({'Lion': 1, 'Cat': 0})

# Define features and target variable
X = data[['voice_loudness', 'size', 'shedding']]
y = data['Lion_or_Cat']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling: Standardize features to help regularization work better
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the logistic regression model with regularization
# Lower values of C increase regularization strength
model = LogisticRegression(C=0.1, penalty='l2')  # C < 1 increases regularization
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy with Regularization: {accuracy:.2f}")

# Example: Predict if a new feline (size=30 cm, loudness=3 dB, shedding=15 hairs per day) is a lion
new_feline = pd.DataFrame({'voice_loudness': [3], 'size': [30], 'shedding': [15]})
new_feline_scaled = scaler.transform(new_feline)
prediction = model.predict(new_feline_scaled)
print("Prediction (1 = Lion, 0 = Cat):", prediction[0])

# Probability output for insight
prediction_prob = model.predict_proba(new_feline_scaled)
print("Prediction Probability (Cat, Lion):", prediction_prob[0])
