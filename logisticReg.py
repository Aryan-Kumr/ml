import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error
from math import sqrt

data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0 = malignant, 1 = benign)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=10000)  # Ensure convergence
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("MSE:", mse)
print("RMSE:", rmse)

# ===================================================== #



import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error
from math import sqrt

data = load_breast_cancer()
X = data.data[:, 0].reshape(-1, 1) 
y = data.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("MSE:", mse)
print("RMSE:", rmse)

# New test data 
new_test_data = np.array([[14.0], [20.0], [10.0], [25.0]]) 

new_pred = model.predict(new_test_data)
new_pred_prob = model.predict_proba(new_test_data)[:, 1]

for i, val in enumerate(new_test_data):
    print(f"Input feature: {val[0]} => Predicted class: {new_pred[i]} (Prob benign: {new_pred_prob[i]:.2f})")


# ------------------------------------------------------------------------------ #



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error
from math import sqrt

data = load_breast_cancer()
X = data.data[:, 0].reshape(-1, 1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("MSE:", mse)
print("RMSE:", rmse)

new_test_data = np.array([[14.0], [20.0], [10.0], [25.0]])

new_pred = model.predict(new_test_data)
new_pred_prob = model.predict_proba(new_test_data)[:, 1]

for i, val in enumerate(new_test_data):
    print(f"Input feature: {val[0]} => Predicted class: {new_pred[i]} (Prob benign: {new_pred_prob[i]:.2f})")

plt.figure(figsize=(10, 6))

plt.scatter(X_train[y_train == 0], y_train[y_train == 0], color='red', label='Malignant (0)', alpha=0.6)
plt.scatter(X_train[y_train == 1], y_train[y_train == 1], color='green', label='Benign (1)', alpha=0.6)

x_values = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_proba = model.predict_proba(x_values)[:, 1]
plt.plot(x_values, y_proba, color='blue', linewidth=2, label='Predicted probability (benign)')

for i, x_val in enumerate(new_test_data):
    plt.scatter(x_val, new_pred_prob[i], color='black', marker='x', s=100,
                label=f'New input: {x_val[0]}' if i == 0 else "")

plt.xlabel('Mean Radius')
plt.ylabel('Probability of Benign')
plt.title('Univariate Logistic Regression: Mean Radius vs Probability of Benign')
plt.legend()
plt.grid(True)
plt.show()
