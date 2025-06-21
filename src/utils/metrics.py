import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def compute_metrics(y_true, y_pred):
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    
    sensitivity = np.mean(TP / (TP + FN + 1e-10))
    specificity = np.mean(TN / (TN + FP + 1e-10))
    score = (sensitivity + specificity) / 2
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average=None)
    
    return sensitivity, specificity, score, accuracy, cm, f1
