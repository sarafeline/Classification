import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

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

# Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Decision Tree Rules
tree_rules = export_text(clf, feature_names=['Ear_Shape', 'Face_Shape', 'Whiskers'])
print(tree_rules)

# Predictions
y_pred = clf.predict(X_train)

# Print Results
print("Predictions:", y_pred)
print("Actual:", y_train)

# Example prediction
example = np.array([[1, 1, 1]])  # Pointy ears, round face shape, whiskers present
prediction = clf.predict(example)
print(f'Is it a cat? {"Yes" if prediction[0] == 1 else "No"}')


Output is:
 
Random forest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data
X_train = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]])

y_train = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0])

# Train-test split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_split, y_train_split)

# Predictions
y_pred = clf.predict(X_test_split)

# Accuracy
accuracy = accuracy_score(y_test_split, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Example prediction
example = np.array([[1, 1, 1]])  # Pointy ears, round face shape, whiskers present
prediction = clf.predict(example)
print(f'Is it a cat? {"Yes" if prediction[0] == 1 else "No"}')
