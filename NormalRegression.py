import numpy as np

'''====================Lasso madel===================='''

def lambda_update(x,y,B,beta,lam,epsilon):
    temp_1=getB(x)
    temp_2=np.matmul(temp_1,beta)
    temp_3=getZ(x,y)

    A_1=2*(temp_2-temp_3)

    B_inv=np.linalg.inv(B)
    A_2=-np.matmul(B_inv,np.sign(beta))

    A=np.matmul(A_1.T,A_2)

    return lam-epsilon*A[0][0]

def ridgeRegression(x,y,beta,B,Z,lam,epsilon):
    row,col=x.shape
    I=np.identity(col)

    new_lam=lambda_update(x,y,B,beta,lam,epsilon)
    new_B=getB_new(x,B)

    A_11=new_B+new_lam*I
    A_1=np.linalg.inv(A_11)

    A_2=getZ_new(x,y,Z)

    return np.matmul(A_1,A_2)




def costFunc_1(x,y,beta,lam):
    A_1=np.matmul(y.T,y)[0][0]
    A_2=np.matmul(np.matmul(beta.T,x.T),np.matmul(x,beta))[0][0]
    A_3=2*np.matmul(np.matmul(y.T,x),beta)[0][0]
    A_4=lam*np.sum(np.abs(beta))

    return A_1+A_2-A_3+A_4

def d_costFunc_(x,y,beta,lam):
    A_1=2*np.matmul(np.matmul(x.T,x),beta)
    A_2=2*np.matmul(x.T,y)
    A_3=lam*np.sign(beta)

    return A_1-A_2+A_3

def dd_costFunc(x,y,beta,lam):
    pass


def griandDecent(x,y,lam):
    row,col=x.shape
    beta=np.zeros(col)

    j=-d_costFunc_(x,y,beta,lam)







'''================Normal Linear Regression==========='''
def getB(x):
    return np.matmul(x.T,x)

def getB_new(x,B):
    return B+getB(x)

def getC(x,B_inv):
    return np.matmul(np.matmul(x,B_inv),x.T)

def getZ(x,y):
    return np.matmul(x.T,y)

def getZ_new(x,y,Z):
    temp=getZ(x,y)
    return Z+temp


def generalHermit(x,B_inv):
    row,col=x.shape
    I=np.identity(row)

    temp=I+np.matmul(np.matmul(x,B_inv),x.T)
    temp_inv=np.linalg.inv(temp)

    return np.matmul(np.matmul(x.T,temp_inv),x)

def B_invxy(B_inv,x,y):
    return np.matmul(np.matmul(B_inv,x.T),y)



#streaming algorithm of normal linear regression
def StreamReg(data,y,beta,B_inv):

    #B_inv=np.linalg.inv(B)

    temp_1 = generalHermit(data,B_inv)
    temp_2 = np.matmul(B_inv, data.T)
    #print(temp_1)
    #print(temp_2)
    #print(temp_2.shape)
    #print(beta)

    A_1=beta

    A_2=np.matmul(np.matmul(B_inv,temp_1),beta)

    A_3=np.matmul(temp_2,y)

    A_41=np.matmul(B_inv,temp_1)
    A_42=np.matmul(temp_2,y)
    A_4=np.matmul(A_41,A_42)

    new_beta=A_1-A_2+A_3-A_4

    return new_beta

def StreamEst(data,y,s2,B_inv,Z):
    row,col=data.shape

    temp_1=np.matmul(B_inv,Z)
    temp_2=B_invxy(B_inv,data,y)
    H=generalHermit(data,B_inv)

    A_1=s2

    A_21=np.matmul(data.T,y)
    A_2=np.matmul(A_21.T,temp_2)

    A_3=2*np.matmul(Z.T,temp_2)

    A_4 =   np.matmul(np.matmul(temp_1.T,H),temp_1)
    A_5 =   np.matmul(np.matmul(temp_2.T, H), temp_2)
    A_6 = 2*np.matmul(np.matmul(temp_1.T, H), temp_2)

    #print(temp_1.shape)
    #print(temp_2.shape)
    #print(A_1.shape)
    #print(A_2.shape)
    #print(A_3.shape)
    #print(A_4.shape)
    #print(A_5.shape)
    #print(A_6.shape)




    new_s2=A_2+A_3-(A_4+A_5+A_6)
    new_s2=A_1+np.matmul(y.T,y)-new_s2

    return new_s2

def Reg(x,y):
    B=getB(x)
    B_inv=np.linalg.inv(B)

    beta=np.matmul(np.matmul(B_inv,x.T),y)

    s2_1=np.matmul(y.T,y)
    s2_2=np.matmul(x.T,y)
    s2_3=np.matmul(np.matmul(s2_2.T,B_inv),s2_2)
    s2=s2_1-s2_3

    return beta,s2

#
#
# def createdata2(beta, sigma, x_min, x_max, n, dim):
#     X = np.zeros(shape=(n, dim))
#     Y = np.zeros(n)
#     for i in range(dim):
#         epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)
#         raw_x = np.random.normal(loc=(x_min + x_max) / 2, scale=(x_min + x_max) ** 0.65, size=100 * n)
#         x = np.random.choice(raw_x, n)
#
#         X[:, i] = x
#         Y += beta[i] * x
#     return X, Y
#
#
# def cal_SSQ2(beta, intercept, X, Y):
#     ssq = 0
#     for (x, y) in zip(X, Y):
#         ssq += (y - np.dot(beta, x)) ** 2
#     return ssq
#
#
#
# import time
# dim = 100
# N = 10000
# k = 100
# n = int(N / k)
#
# X = np.zeros(shape=(n, dim))
# Y = np.zeros(n)
#
# beta_con = np.random.randint(1, 5, size=dim)
# sigma_con = 10
#
# a = 1
#
# x_min = 0
# x_max = 100
#
# SSQ1 = []
# timecost = []
#
# for i in range(k):
#     print(i)
#     X_temp, Y_temp = createdata2(beta_con, sigma_con, x_min, x_max, n, dim)
#     start = time.time()
#     if a == 1:
#         X = X_temp
#         Y = Y_temp
#         a = 0
#     else:
#         X = np.row_stack((X, X_temp))
#         Y = np.row_stack((Y, Y_temp))
#
#     beta, s2 = Reg(X, Y)
#     SSQ1.append(s2)
#
#     end = time.time()
#     timecost.append(end - start)
#
#
#


'''

#debugging

x1=np.array([[1],[2],[3],[4],[5],[6],[7]])
y1=np.array([1,2,3,4,5,6,7])

x1=np.array([[8],[9],[10],[11],[12],[13],[14]])
y1=np.array([8,9,10,11,12,13,14])

beta,s2=Reg(x1,y1)
print(beta)
print(s2)

beta,s2=reg.Reg(x1,y1)

B=reg.getB(x1)
Z=reg.getZ(x1,y1)
B_inv=np.linalg.inv(B)
num=x1.shape[0]

C=reg.getC(x2,B_inv)
new_beta=reg.StreamReg(x2,y2,beta,B_inv)
new_s2=reg.StreamEst(x2,y2,C,s2,num,B_inv,Z)
'''