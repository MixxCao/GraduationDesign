import numpy as np








def g_inv(x):
    #print('ha')
    ex=np.exp(x)
    return ex/(1+ex)

def updateS(beta,x,y,s_zz,s_xz,s_xx):
    #print(x)
    #print(beta)
    eta=np.dot(beta,x)
    #print(eta)

    mu=g_inv(eta)

    w=mu*(1-mu)
    #print(mu)
    z=eta+(y-mu)/w


    s_zz=s_zz+w*z*z

    #print(s_xz)
    #print(x)
    s_xz=s_xz+w*z*x

    c=np.mat(x)
    #print(s_xx)
    #print(w)
    s_xx=s_xx+w*np.array(np.matmul(c.T,c))

    return s_zz,s_xz,s_xx

def update(beta,X,y,s_zz,s_xz,s_xx):
    #print('1')

    for x,y_1 in zip(X,y):
        #print(x)
        #print(beta)
        s_zz,s_xz,s_xx=updateS(beta, x, y_1, s_zz, s_xz, s_xx)

    #print('2')
    s_xx_inv=np.linalg.pinv(s_xx)
    #print(s_xx)
    beta=np.matmul(s_xx_inv,s_xz)
    #print('3')
    a=beta
    #print(a.shape)

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
        #print(s_zz)
        #print(S_zz)

        S_zz.append(s_zz)
        S_xz.append(s_xz)
        S_xx.append(s_xx)

    return beta,nsigma,S_zz,S_xz,S_xx




'''

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X_raw= iris.data
y_raw = iris.target
y=y_raw[0:100]
X=X_raw[0:100]

x_l=[X[0:60],X[60:80]]
y_l=[y[0:60],y[60:80]]

max_iter=20

clf_1 = LogisticRegression(random_state=0, C=1e5,solver='sag',max_iter=2000,multi_class='ovr')
ini_model=clf_1.fit(x_l[0],y_l[0])
b=ini_model.coef_

beta=b[0]
nsigma=0

S_zz=[]
S_xz=[]
S_xx=[]

beta_1, nsigma, S_zz, S_xz, S_xx = streamingIRWLS_init(beta,x_l[0],y_l[0],S_zz,S_xz,S_xx,max_iter)

#print(S_zz)
beta_2, nsigma, S_zz, S_xz, S_xx = streamingIRWLS(beta_1, x_l[1], y_l[1], S_zz, S_xz, S_xx,max_iter)

print(beta_1)


'''

