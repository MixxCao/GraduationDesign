import numpy as np
import math

def dist1D(x,y):
    return np.sqrt(np.sum((y-x)**2))

def dist2D(x,y):
    return np.sqrt(np.sum((y - x) ** 2, axis=1))

def distset(x,mdn):
    return np.min(dist2D(x,mdn))

def assign(data,mdn):
    asgn = np.array([])
    a, b = mdn.shape
    fnum = np.zeros(a)

    for x in data:
        index = int(np.argmin(dist2D(x, mdn)))
        asgn = np.append(asgn, index)
        fnum[index] += 1

    deltlist=np.array([])
    for k,test in enumerate(fnum):
        if test==0:
            np.append(deltlist,k)
    mdn = np.delete(mdn, deltlist, axis=0)
    fnum = np.delete(fnum, deltlist, axis=0)
    return mdn, asgn, fnum


def SSQ(data,mdn,asgn):
    sm=0
    for k,x in enumerate(data):
        dist=dist1D(x,mdn[int(asgn[k])])
        sm += dist**2
    return sm

def cost(data,mdn,asgn,z):
    return len(mdn)*z+SSQ(data,mdn,asgn)

def gain(x,data,mdn,asgn,z):
    cst=cost(data,mdn,asgn,z)

    temp_mdn=np.row_stack((mdn,x))
    temp_mdn,temp_asgn,temp_fnum=assign(data,temp_mdn)

    temp_cst=cost(data,temp_mdn,temp_asgn,z)

    return cst-temp_cst

def InitialSolution(data, z):
    np.random.shuffle(data)
    mdn=np.array([data[0]])
    for x in data:
        d=distset(x,mdn)
        if d>=z:
            mdn=np.row_stack((mdn,x))
        else:
            ran=np.random.uniform(0,z)
            if ran<=d:
                mdn=np.row_stack((mdn,x))
            else:
                pass

    return mdn

def FL(data,z,epsilon,mdn):
    mdn,asgn,fnum=assign(data,mdn)
    cst=cost(data,mdn,asgn,z)

    while True:
        for x in data:
            gainx = gain(x, data, mdn, asgn, z)
            if gainx > 0:
                mdn = np.row_stack((mdn, x))
                mdn, asgn, fnum = assign(data, mdn)
            else:
                pass

        new_cst = cost(data, mdn, asgn, z)
        if (new_cst > (1 - epsilon) * cst):
                return mdn
        else:
            cst = new_cst

def LSEARCH(data,k,epsilon,epsilon2, epsilon3):
    zmin = 0
    zmax = np.sum(dist2D(data[0], data))
    z=zmax/2+zmin/2

    mdn=InitialSolution(data,z)
    while True:
        mdn = FL(data, z, epsilon, mdn)  # 认为about以为着差一两个，可以修改
        if len(mdn) > k:
            zmin = z
        else:
            zmax = z

        if len(mdn) == k:
            return mdn
        elif zmin > (1 - epsilon3) * zmax:
            return mdn
        else:
            pass
        z = (zmin + zmax) / 2




def w_assign(data,weigh,mdn):
    asgn = np.array([])
    a, b = mdn.shape
    fnum = np.zeros(a)

    for w,x in zip(weigh,data):
        index = int(np.argmin(dist2D(x, mdn)))
        asgn = np.append(asgn, index)
        fnum[index] += w

    deltlist = np.array([])
    for k, test in enumerate(fnum):
        if test == 0:
            np.append(deltlist, k)
    mdn = np.delete(mdn, deltlist, axis=0)
    fnum = np.delete(fnum, deltlist, axis=0)
    return mdn,fnum


def reunion(mdn,fnum,k):
    epsilon = 0.1
    epsilon2 = 0.1
    epsilon3 = 0.01

    new_mdn = w_LSEARCH(mdn,fnum, k, epsilon, epsilon2, epsilon3)
    mdn,fnum=w_assign(mdn,fnum,new_mdn)

    return mdn,fnum






def CLU(data,k,mdn,fnum):
    epsilon=0.01
    epsilon2=0.001
    epsilon3=0.01

    new_mdn=LSEARCH(data,k,epsilon,epsilon2, epsilon3)
    new_mdn,new_asg,new_fnum=assign(data,new_mdn)

   # mdn=np.row_stack((mdn,new_mdn))
   # fnum=np.append(fnum,new_fnum)

   # mdn,fnum=reunion(mdn,fnum,k)

    return new_mdn,new_fnum

    #return mdn,fnum



def InitialSolution2(data, z):
    #np.random.shuffle(data)
    mdn=np.array([data[0]])
    for x in data:
        d=distset(x,mdn)
        if d>=z:
            mdn=np.row_stack((mdn,x))
        else:
            ran=np.random.uniform(0,z)
            if ran<=d:
                mdn=np.row_stack((mdn,x))
            else:
                pass

    return mdn


def w_gain(x,weigh,data,mdn,asgn,z):
    cst=w_cost(data,weigh,mdn,asgn,z)

    temp_mdn=np.row_stack((mdn,x))
    temp_mdn,temp_asgn,temp_fnum=assign(data,temp_mdn)

    temp_cst=w_cost(data,weigh,temp_mdn,temp_asgn,z)

    return cst-temp_cst


def w_SSQ(data,weigh,mdn,asgn):
    sm=0
    for k,x in enumerate(data):
        #print(k)
        #print(x)
        dist=dist1D(x,mdn[int(asgn[k])])
        sm += (dist**2)*weigh[k]
    return sm



def w_cost(data,weigh,mdn,asgn,z):
    return len(mdn)*z+w_SSQ(data,weigh,mdn,asgn)

def w_FL(data,weigh,z,epsilon,mdn):
    mdn,asgn,fnum=assign(data,mdn)
    cst=w_cost(data,weigh,mdn,asgn,z)

    while True:
        for x in data:
            gainx = gain(x, data, mdn, asgn, z)
            if gainx > 0:
                mdn = np.row_stack((mdn, x))
                mdn, asgn, fnum = assign(data, mdn)
            else:
                pass

        new_cst = cost(data, mdn, asgn, z)
        if (new_cst > (1 - epsilon) * cst):
                return mdn
        else:
            cst = new_cst


def w_LSEARCH(data,weigh,k,epsilon,epsilon2, epsilon3):
    zmin = 0
    zmax = np.sum(dist2D(data[0], data))
    z=zmax/2+zmin/2

    mdn=InitialSolution(data,z)
    while True:
        mdn = w_FL(data, weigh,z, epsilon, mdn)  # 认为about以为着差一两个，可以修改
        if len(mdn) > k:
            zmin = z
        else:
            zmax = z

        if len(mdn) == k:
            return mdn
        elif zmin > (1 - epsilon3) * zmax:
            return mdn
        else:
            pass
        z = (zmin + zmax) / 2

def test_SSQ(data,mdn):
    sm=0
    for x in data:
        index=int(np.argmin(distset(x,mdn)))
        sm+=dist1D(x,mdn[index])**2

    return sm



