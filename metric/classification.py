import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))
    return tp / (tp + fp)

def recall(y_true, y_pred, label):
    tp = np.sum((y_true == label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))
    return tp / (tp + fn)

def f1_score(y_true, y_pred, label):
    prec = precision(y_true, y_pred, label)
    rec = recall(y_true, y_pred, label)
    return 2 * (prec * rec) / (prec + rec)

def confusion_matrix(num_classes, y_true, y_pred):

    mat = np.zeros((num_classes, num_classes))
    dim = len(y_true)
    for i in range(dim):
        mat[y_true[i], y_pred[i]] += 1
    
    return mat