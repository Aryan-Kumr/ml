import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,f1_score, roc_curve, auc)

#disease diagnosis (1 = disease, 0 = healthy)
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # actual
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  # predicted
y_proba = [0.95, 0.1, 0.88, 0.45, 0.05, 0.97, 0.7, 0.2, 0.91, 0.15]  # predicted probabilities

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:\n{cm}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def specificity(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def npv(tn, fn):
    return tn / (tn + fn) if (tn + fn) > 0 else 0

manual_accuracy = accuracy(tp, tn, fp, fn)
manual_precision = precision(tp, fp)
manual_recall = recall(tp, fn)
manual_specificity = specificity(tn, fp)
manual_f1 = f1(tp, fp, fn)
manual_npv = npv(tn, fn)

print("\n--- Manual Metrics ---")
print(f"Accuracy: {manual_accuracy:.4f}")
print(f"Precision: {manual_precision:.4f}")
print(f"Recall (Sensitivity): {manual_recall:.4f}")
print(f"Specificity: {manual_specificity:.4f}")
print(f"F1-Score: {manual_f1:.4f}")
print(f"NPV: {manual_npv:.4f}")

print("\n--- Sklearn Metrics ---")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall (Sensitivity): {recall_score(y_true, y_pred):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")
print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
print(f"NPV: {tn / (tn + fn):.4f}")

fpr, tpr, _ = roc_curve(y_true, y_proba)
auc_score = auc(fpr, tpr)


plt.plot(fpr, tpr, label=f'Model AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], ':', color='gray')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"\nAUC (Model): {auc_score:.4f}")
