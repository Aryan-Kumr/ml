import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Function to evaluate Random Forest with varying number of trees
def evaluate_rf(n_trees_list):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for n_trees in n_trees_list:

        rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
 
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return accuracies, precisions, recalls, f1_scores

# 7. Experiment with different numbers of trees (e.g., 10, 50, 100, 200)
n_trees_list = [10, 50, 100, 200]
accuracies, precisions, recalls, f1_scores = evaluate_rf(n_trees_list)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(n_trees_list, accuracies, marker='o')
plt.title('Accuracy vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(n_trees_list, precisions, marker='o')
plt.title('Precision vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Precision')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(n_trees_list, recalls, marker='o')
plt.title('Recall vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Recall')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(n_trees_list, f1_scores, marker='o')
plt.title('F1-Score vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('F1-Score')
plt.grid(True)

plt.tight_layout()
plt.show()
