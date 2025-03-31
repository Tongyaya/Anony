from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F


def get_onehot_org(y,baseclass = None):
    '''
    Input:
        y: array [N,1]
    Output:
        [N, K-1]
    '''
    idx = np.squeeze(y)
    ss = len(y)
    nclass = len(np.unique(y))
    z = np.zeros([ss,nclass])
    z[np.arange(ss),idx] = 1  
    ls_class = list(np.arange(nclass))
    if baseclass is None:
        baseclass = K-1
    _ = ls_class.pop(baseclass)
    return z[:,ls_class]

def logit_predict(X,beta_hat,Ns0):
     # weighted probs
    link_mu = X@beta_hat  ## N*（K-1）
    prob_hat = np.exp(link_mu);
    prob_hat = np.concatenate((np.ones([prob_hat.shape[0],1]),prob_hat),1)
    prob_hat = prob_hat/np.sum(prob_hat,axis = 1,keepdims=True) 
    prob_hat[np.isnan(prob_hat)] = 0
    
    Ns0_per = Ns0/np.sum(Ns0)
    prob_hat1 = prob_hat/Ns0_per.reshape([1,-1]) # weighted probs
    Yhat = np.argmax(prob_hat1,1) 
    
    return prob_hat, Yhat

def test_accw(X, Y, Ns0, beta_hat):
    prob_hat, Yhat = logit_predict(X,beta_hat,Ns0)
    
    accs = np.zeros(prob_hat.shape[1])
    for k in range(prob_hat.shape[1]):
        accs[k] = np.mean(Yhat[Y==k]==Y[Y==k])
    
    aucs = np.zeros(prob_hat.shape[1])
    Y_1hot = get_onehot_org(Y,0)
    Y_1hot = np.concatenate((1-np.sum(Y_1hot,1).reshape([-1,1]),Y_1hot),1)
    for k in range(prob_hat.shape[1]):
        aucs[k] = roc_auc_score(Y_1hot[:,k], prob_hat[:,k])
    
    print(f'9 cls acc = {np.round(np.mean(accs),3)},auc = {np.round(np.mean(aucs),3)}')


def Params(loss_type):
    if loss_type == "FL":
        params = {
            "input_size":512,
            "num_classes":9,
            "beta":0,
            "gamma":2,
            "bs":32,
            "num_epochs":90,
            "warm_up1":30,
            "warm_up2":60,
            "lr":0.1,
            "multi_lr":0.1             
                 }

    elif loss_type == "CBL":
        params = {
            "input_size":512,
            "num_classes":9,
            "beta":0.9999,
            "gamma":0,
            "bs":1024,
            "num_epochs":90,
            "warm_up1":30,
            "warm_up2":60,
            "lr":0.1,
            "multi_lr":0.1             
                 }
        
    elif loss_type == "CSL":
        params = {
            "input_size":512,
            "num_classes":9,
            "beta":1,
            "gamma":2,
            "bs":2048,
            "num_epochs":90,
            "warm_up1":30,
            "warm_up2":60,
            "lr":0.1,
            "multi_lr":0.1             
                 }
        
    elif loss_type == "RDS":
        params = {
            "input_size":512,
            "num_classes":9,
            "beta":0,
            "gamma":0,
            "bs":256,
            "num_epochs":90,
            "warm_up1":30,
            "warm_up2":60,
            "lr":0.1,
            "multi_lr":0.1             
                 }
    return params
        
def count_unique_elements(tensor):
    unique_values, counts = torch.unique(tensor, return_counts=True)
    result = [ count.item() for value, count in zip(unique_values, counts)]
    return result


def feature_std(X):
    X_non = X[:,1:] 
    Xmax = np.max(X_non,0).reshape([1,-1]); Xmin = np.min(X_non,0).reshape([1,-1])
    idx = np.where(Xmax-Xmin==0)[1]
    X_std = (X_non-Xmin)/(Xmax-Xmin); X_std[:,idx] = 0
    X[:,1:] = X_std
    del X_non, X_std
    return X