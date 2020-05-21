import numpy as np

import sys









def g_inv(x):
    ex=np.exp(x)
    return ex/(1+ex)

def updateS(beta,x,y,s_zz,s_xz,s_xx):

    eta=np.dot(beta,x)

    mu=g_inv(eta)

    w=mu*(1-mu)
    if w==0:
        z=eta+(y-mu)*sys.maxsize
    else:
        z = eta + (y - mu)/w


    s_zz=s_zz+w*z*z


    s_xz=s_xz+w*z*x


    c=np.mat(x)

    s_xx=s_xx+w*np.array(np.matmul(c.T,c))

    return s_zz,s_xz,s_xx

def update(beta,X,y,s_zz,s_xz,s_xx):

    for x,y_1 in zip(X,y):

        s_zz,s_xz,s_xx=updateS(beta, x, y_1, s_zz, s_xz, s_xx)


    s_xx_inv=np.linalg.pinv(s_xx)

    beta=np.matmul(s_xx_inv,s_xz)


    sigma_1=s_zz
    sigma_2=np.matmul(s_xx_inv,s_xz)
    sigma_3=np.dot(sigma_2,s_xz)
    nsigma=sigma_1-sigma_3

    return beta,nsigma,s_zz,s_xz,s_xx

def IRWLS(beta,X,y,s_zz,s_xz,s_xx):

    max_iter=2000
    epsilon=0.001
    k=0
    nsigma=0

    while(k<max_iter):
        beta_new, nsigma, s_zz, s_xz, s_xx=update(beta,X,y,s_zz,s_xz,s_xx)
        #print(k)

        test_vec=beta_new-beta
        test_val=np.sum(np.abs(test_vec))/np.sum(np.abs(beta))
        #print(test_val)
        #print(beta_new)
        beta=beta_new
        if test_val<epsilon:
            break


        k+=1

    return beta, nsigma, s_zz, s_xz, s_xx



def streamingIRWLS(beta,X,y,S_zz,S_xz,S_xx,max_iter):

    nsigma = 0

    for i in range(max_iter):
        s_zz = S_zz[i]
        s_xz = S_xz[i]
        s_xx = S_xx[i]
        beta, nsigma, s_zz, s_xz, s_xx=update(beta,X,y,s_zz,s_xz,s_xx)

        S_zz[i] = s_zz
        S_xz[i] = s_xz
        S_xx[i] = s_xx

    return beta,nsigma,S_zz,S_xz,S_xx

def streamingIRWLS_init(beta,X,y,S_zz,S_xz,S_xx,max_iter):
    nsigma = 0

    n, col = X.shape
    s_zz = 0
    s_xz = np.zeros(col)
    s_xx = np.zeros((col, col))

    for i in range(max_iter):

        beta, nsigma, s_zz, s_xz, s_xx=update(beta,X,y,s_zz,s_xz,s_xx)



        S_zz.append(s_zz)
        S_xz.append(s_xz)
        S_xx.append(s_xx)

    return beta,nsigma,S_zz,S_xz,S_xx







'''
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


path='/Users/mixxxxx/Downloads/DataSet/dota2Dataset/data_withmax.csv'
raw_data=pd.read_csv(path)

data=raw_data.values
data=data.astype(np.float64)

raw_y=data[:,1]
y=np.zeros(len(raw_y))
for i in range(len(y)):
    if raw_y[i]:
        y[i]=1
    else:
        y[i]=0
X=data[:,2:]

Xtmp=np.column_stack((X[:,0:3],X[:,5:8]))
X=np.column_stack((Xtmp,X[:,10]))

X_stan=preprocessing.scale(X)
y_stan=preprocessing.scale(y)

#7:3划分集合
testtrain=int(X.shape[0]*0.7)
Xtrain=X[:testtrain,:]
Xtest=X[testtrain:,:]
ytrain=y[:testtrain]
ytest=y[testtrain:]

Xtrain_stan=X_stan[:testtrain,:]
Xtest_stan=X_stan[testtrain:,:]

k=100
X1=Xtrain[0:k,:]
y1=ytrain[0:k]


clf = LogisticRegression(random_state=1, C=1e5,solver='sag',max_iter=2000,multi_class='ovr')
model = clf.fit(X1, y1)


beta=model.coef_[0]
nsigma=0
max_iter=5

beta_list=[]
S_zz=[]
S_xz=[]
S_xx=[]
timelist=[]

beta, nsigma, S_zz, S_xz, S_xx = streamingIRWLS_init(beta,X1,y1,S_zz,S_xz,S_xx,max_iter)

'''