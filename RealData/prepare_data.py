import numpy as np
from utils import *


def mkdata(loss_type,traindata,testdata,params):
    torch.manual_seed(42)
    X = traindata['X0']; Y = traindata['Y0']; Ns0 = traindata['Ns0']
    X1 = testdata['X1']; Y1 = testdata['Y1']; Ns1 = testdata['Ns1']

    X = np.concatenate((np.ones([X.shape[0],1]),X),1)
    X1 = np.concatenate((np.ones([X1.shape[0],1]),X1),1)
    Y = Y.astype(np.int16).reshape(-1)
    Y1 = Y1.astype(np.int16).reshape(-1)
    X = feature_std(X); X1 = feature_std(X1)

    N = X.shape[0]  
    p = X.shape[1]     
    K = 9    
    baseclass = 0
    K0_pval = np.sum(Y == 0)/ Y.shape[0]
    idx_0 = np.where(Y==0)[0]  
    if loss_type == "RDS":
        pi = 0.25
        idx_0 = np.where(Y==0)[0]  
        np.random.seed(0)
        idx_0_sub = np.random.choice(idx_0,int(idx_0.shape[0]*pi),replace=False) 
        idx_others = np.where(Y!=0)[0] 
        idx = np.concatenate((idx_0_sub,idx_others)) 
        idx = np.sort(idx)
        X = X[idx]; Y = Y[idx]; N = X.shape[0]  
    X = X[:,1:]; X1 = X1[:,1:] 

    X_torch = torch.tensor(X,dtype=torch.float32)
    Y_torch = torch.tensor(Y,dtype=torch.long)
    samples_per_cls = count_unique_elements(Y_torch)
    dataset = TensorDataset(X_torch, Y_torch)
    dataloader = DataLoader(dataset, batch_size=params["bs"], shuffle=True)
    
    X1_torch = torch.tensor(X1,dtype=torch.float32)
    Y1_torch = torch.tensor(Y1,dtype=torch.long)

    
    return samples_per_cls,dataloader,Ns0,X1_torch,Y1_torch,X1,Y1
