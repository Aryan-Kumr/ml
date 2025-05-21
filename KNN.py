import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    'Pregnancies': [2, 4, 3, 1, 5, 0, 6, 2, 3, 1],
    'Glucose': [85, 150, 120, 95, 180, 70, 200, 130, 145, 100],
    'BloodPressure': [66, 88, 72, 80, 90, 65, 85, 70, 75, 68],
    'SkinThickness': [29, 35, 25, 20, 45, 18, 50, 30, 33, 27],
    'Insulin': [0, 130, 85, 90, 200, 70, 250, 100, 140, 120],
    'BMI': [26.6, 33.2, 28.1, 30.5, 40.1, 25.4, 45.0, 31.2, 36.5, 29.3],
    'DiabetesPedigreeFunction': [0.351, 0.672, 0.500, 0.245, 1.200, 0.198, 1.500, 0.520, 0.801, 0.300],
    'Age': [25, 45, 35, 22, 50, 21, 60, 33, 40, 28],
    'Outcome': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy with K=3 (Euclidean): {acc:.4f}")

k_values = list(range(1, len(X_train) + 1)) 
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    
# Plot accuracy vs K
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K Value')
plt.grid(True)
plt.show()

# Best K
best_k = k_values[np.argmax(accuracies)]
print(f"Best K value: {best_k}")

# Manhattan distance
knn = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')
knn.fit(X_train, y_train)
y_pred_manhattan = knn.predict(X_test)
acc_manhattan = accuracy_score(y_test, y_pred_manhattan)
print(f"Accuracy with K={best_k} (Manhattan): {acc_manhattan:.4f}")
