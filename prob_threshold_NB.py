from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the iris dataset (or any other dataset you want to use)
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize Gaussian Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict probabilities for test set
probabilities = model.predict_proba(X_test)

# Get the top 3 predicted classes for each sample
top_3_classes = np.argsort(probabilities, axis=1)[:, -3:]

# Calculate the probability thresholds for top 3 classes
threshold = 0.95
top_3_probs = np.sort(probabilities, axis=1)[:, -3:]
class_predictions = np.where(top_3_probs[:, -1] > threshold, top_3_classes[:, -1],
                              np.where(top_3_probs[:, -2] > threshold, top_3_classes[:, -2],
                                       np.where(top_3_probs[:, -3] > threshold, top_3_classes[:, -3], -1)))

# Evaluate the model
accuracy = accuracy_score(y_test, class_predictions)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, class_predictions))
