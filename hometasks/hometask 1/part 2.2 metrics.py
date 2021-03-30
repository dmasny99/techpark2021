import numpy as np

def classifier(y_predict,threshold,q):
    y_calculated = np.zeros(y_predict[:q].shape[0])
    for i in range(y_calculated.shape[0]):
        for j in range(y_predict[:q].shape[1]):
            if y_predict[i][j] >= threshold:
                y_calculated[i] = j
    return y_calculated

def accuracy_score(y_true,y_predict, percent = None):
    threshold = 1/y_predict.shape[1]
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    y_calculated = classifier(y_predict,threshold,q)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(y_true[:q].shape[0]):
        if (y_true[i] == 1) and (y_calculated[i] == 0):
            FN = FN + 1
        if (y_true[i] == 1) and (y_calculated[i] == 1):
            TP = TP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 1):
            FP = FP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 0):
            TN = TN + 1
    return ((TP+TN)/(TP+TN+FP+FN))

def precision_score(y_true, y_predict, percent=None):
    threshold = 1/y_predict.shape[1]
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    y_calculated = classifier(y_predict,threshold,q)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(y_true[:q].shape[0]):
        if (y_true[i] == 1) and (y_calculated[i] == 0):
            FN = FN + 1
        if (y_true[i] == 1) and (y_calculated[i] == 1):
            TP = TP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 1):
            FP = FP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 0):
            TN = TN + 1
    return TP/(TP+FP)

def recall_score(y_true, y_predict, percent=None):
    threshold = 1/y_predict.shape[1]
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    y_calculated = classifier(y_predict,threshold,q)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(y_true[:q].shape[0]):
        if (y_true[i] == 1) and (y_calculated[i] == 0):
            FN = FN + 1
        if (y_true[i] == 1) and (y_calculated[i] == 1):
            TP = TP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 1):
            FP = FP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 0):
            TN = TN + 1
    return TP/(TP+FN)

def lift_score(y_true, y_predict, percent=None):
    threshold = 1/y_predict.shape[1]
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    y_calculated = classifier(y_predict,threshold,q)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(y_true[:q].shape[0]):
        if (y_true[i] == 1) and (y_calculated[i] == 0):
            FN = FN + 1
        if (y_true[i] == 1) and (y_calculated[i] == 1):
            TP = TP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 1):
            FP = FP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 0):
            TN = TN + 1
    
    return (precision_score(y_true, y_predict, percent)/((TP+FN)*(TP+TN+FP+FN)))

def f1_score(y_true, y_predict, percent=None):
    threshold = 1/y_predict.shape[1]
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    y_calculated = classifier(y_predict,threshold,q)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(y_true[:q].shape[0]):
        if (y_true[i] == 1) and (y_calculated[i] == 0):
            FN = FN + 1
        if (y_true[i] == 1) and (y_calculated[i] == 1):
            TP = TP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 1):
            FP = FP + 1
        if (y_true[i] == 0) and (y_calculated[i] == 0):
            TN = TN + 1
    p = precision_score(y_true, y_predict, percent)
    r = recall_score(y_true, y_predict, percent)
    return (2*p*r/(p+r))
