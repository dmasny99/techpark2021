import numpy as np

def classifier(y_predict,threshold,q):
    y_calculated = np.zeros(y_predict[:q].shape[0])
    for i in range(y_calculated.shape[0]):
        for j in range(y_predict[:q].shape[1]):
            if y_predict[i][j] >= threshold:
                y_calculated[i] = j
    return y_calculated

def conf_mat(y_true,y_predict, percent = None):
    threshold = 1/y_predict.shape[1]
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    y_calculated = classifier(y_predict,threshold,q)
    print(type(int(y_calculated[1])))
    c_m = np.zeros((len(np.unique(y_true[:q])),len(np.unique(y_true[:q]))))
    for i in range(y_true[:q].shape[0]):
        c_m[int(y_calculated[i])][int(y_true[i])] += 1 
    return c_m

def accuracy_score(y_true,y_predict, percent = None):
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    conf_matrix =conf_mat(y_true,y_predict, percent)
    matched_classes = 0
    for j in range(conf_matrix.shape[1]):
        matched_classes += conf_matrix[j][j]
    return (matched_classes/np.sum(conf_matrix))

def precision_score(y_true, y_predict, percent=None):
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    conf_matrix =conf_mat(y_true,y_predict, percent)
    prec_sc_mat = np.zeros(conf_matrix.shape[0])
    for i in range(conf_matrix.shape[0]):
        prec_sc_mat[i] = conf_matrix[i][i]/np.sum(conf_matrix[i])
    if y_predict.shape[1] == 2:
        return prec_sc_mat[1]
    else:
        return prec_sc_mat
def recall_score(y_true, y_predict, percent=None):
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    conf_matrix =conf_mat(y_true,y_predict, percent)
    rec_sc_mat = np.zeros(conf_matrix.shape[0])
    for i in range(conf_matrix.shape[0]):
        rec_sc_mat[i] = conf_matrix[i][i]/np.sum(conf_matrix[:,i])
    if y_predict.shape[1] == 2:
        return rec_sc_mat[1]
    else:
        return rec_sc_mat
def lift_score(y_true, y_predict, percent=None):
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    conf_matrix =conf_mat(y_true,y_predict, percent)
    lift_sc_mat = np.zeros(conf_matrix.shape[0])
    if y_predict.shape[1] == 2:
        lift = precision_score(y_true, y_predict, percent)/(np.sum(conf_matrix[:,1])/np.sum(conf_matrix))
        return round(lift,3)
    else:
        for i in range(conf_matrix.shape[0]):
            lift_sc_mat[i] = round((precision_score(y_true, y_predict, percent)[i]/((np.sum(conf_matrix[:,i])/np.sum(conf_matrix)))),3)
        return lift_sc_mat
def f1_score(y_true, y_predict, percent=None):
    q = y_true.shape[0]
    if percent is not None:
        q = round(percent * y_true.shape[0]/100)
    conf_matrix =conf_mat(y_true,y_predict, percent)
    if y_predict.shape[1] == 2:
        p = precision_score(y_true, y_predict, percent)
        r = recall_score(y_true, y_predict, percent)
        f1_sc = round((2*p*r/(p+r)),3)
        return f1_sc
    else:
        f1_sc_mat = np.zeros(conf_matrix.shape[0])
        for i in range(conf_matrix.shape[0]):
            p_i = precision_score(y_true, y_predict, percent)[i]
            r_i = recall_score(y_true, y_predict, percent)[i]
            f1_sc_mat[i] = round((2*p_i*r_i/(p_i+r_i)),3)
        return f1_sc_mat
