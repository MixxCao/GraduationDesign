import numpy as np


def accuracyRate(X, y, tree):
    count = 0
    pre = tree.predict(X)

    for p, yy in zip(pre, y):
        if p == yy:
            count += 1
    return count / X.shape[0]


def purneTree(X, y, treelist, cap):
    if len(treelist) <= cap:
        return treelist

    accuracyRatelist = []
    for tree in treelist:
        accuracyRatelist.append(accuracyRate(X, y, tree))

    dlnum = np.argmin(accuracyRatelist)
    del treelist[dlnum]
    return treelist


def ensemblepredict(X, Y, treelist):
    prelist = np.ones(len(Y)) * 3
    # finpre=np.zeros(len(Y))
    for tree in treelist:
        pre = tree.predict(X)
        prelist = np.column_stack((prelist, pre))

    count = 0
    for p, y in zip(prelist, Y):
        counts = np.bincount(p.astype(int))
        pp = np.argmax(counts)
        if pp == y:
            count += 1
        else:
            pass
    return count / len(Y)
