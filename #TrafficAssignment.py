# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
-------------------------------------------------

"""

import numpy as np
import pandas as pd
from sympy import *


def bisection(func):
    left = 0
    right = 1
    eps = 10e-3
    mid = (left + right) / 2
    i = 0
    # while abs (func (mid)) > eps:
    while i<5:
        mid = (left + right) / 2
        if func (left) * func (mid) <= 0:
            right = mid
        else:
            left = mid
        print (mid)
        i = i + 1
    return mid


def BPR(x, t0, c, a=0.15, b=4):
    '''
    link travel time function
    the Bureau of Public Roads
    t=t0(1+a(x/c)^b)
    :param x: traffic flow on link (i,j)
    :param t0: free-flow travel time on link (i,j)
    :param c: flow capacity on link (i,j)
    :param a,b: BPR function parameters
    :return: t,travel time on link (i,j)
    '''
    t = t0 * (1 + a * pow ((x / c), b))
    return t


def LN(x, b, k):
    '''
    t=b+kx
    '''
    t = x * k + b
    return t


def OBJ_UE_BPR(x, t0, c, a=0.15, b=4):
    w = symbols ('w')
    z = integrate (BPR (w, t0, c), (w, 0, x))
    return z


def OBJ_UE_LN(x, b, k):
    w = symbols ('w')
    z = integrate (LN (w, b, k), (w, 0, x))
    return z


def AN_UE_LN(x):
    q = 8
    b = np.array ([4, 8])
    k = np.array ([8, 2])
    time = np.vectorize (LN)
    obj = np.vectorize (OBJ_UE_LN)
    t = time (x, b, k)
    y = np.zeros (len (x))
    y[np.argmin (t)] = 1
    y = y * q
    z = obj (x, b, k)
    z = sum (z)

    def df_obj(alpha):
        df = (y - x) * time (x + alpha * (y - x), b, k)
        df = sum (df)
        return df

    alpha = bisection (df_obj)
    x1 = x + alpha * (y - x)
    conv = sqrt (sum (pow (x1 - x, 2))) / sum (x)
    return t, y, x1, z, alpha, conv


def AN_SO_LN(x):
    q = 8
    b = np.array ([4, 8])
    k = np.array ([8, 2])
    time = np.vectorize (LN)
    obj = np.vectorize (OBJ_UE_LN)
    t = time (x, b, k)
    t_an = t + k * x
    y = np.zeros (len (x))
    y[np.argmin (t_an)] = 1
    y = y * q
    z = obj (x, b, k)
    z = sum (z)

    def df_obj(alpha):
        df = x * (alpha * (y - x))
        df = sum (df)
        return df

    alpha = bisection (df_obj)
    x1 = x + alpha * (y - x)
    conv = sqrt (sum (pow (x1 - x, 2))) / sum (x)
    return t, y, x1, z, alpha, conv


def AN_UE_BPR(x):
    '''
    all or nothing for user eq
    :return: y, auxiliary flow
    '''
    q = 10
    t0 = np.array ([10, 20, 25])
    c = np.array ([2, 4, 3])
    time = np.vectorize (BPR)
    obj = np.vectorize (OBJ_UE_BPR)
    t = time (x, t0, c)
    y = q * (t == np.min (t))
    z = obj (x, t0, c)
    z = sum (z)

    def df_obj(alpha):
        df = (y - x) * time (x + alpha * (y - x), t0, c)
        df = sum (df)
        return df

    alpha = bisection (df_obj)
    x1 = x + alpha * (y - x)
    return t, y, x1, z, alpha


def test_UE_BPR():
    # parameter
    q = 10
    t0 = np.array ([10, 20, 25])
    c = np.array ([2, 4, 3])
    # initialization
    x = np.array ([0, 0, 0])
    time = np.vectorize (BPR)
    t = time (x, t0, c)
    x = q * (t == np.min (t))
    # update
    out = []
    for i in range (6):
        t, y, x, z, alpha = AN_UE_BPR (x)
        t = np.round (t, 1)
        x = np.round (x, 2)
        out.append ([i + 1, t, y, x, round (z, 2), round (alpha, 4)])
    out = pd.DataFrame (out, columns=['step', 'time', 'auxiliary', 'flow', 'obj', 'alpha'])
    print (out)


def test_UE_LN():
    # parameter
    q = 8
    b = np.array ([4, 8])
    k = np.array ([8, 2])

    # initialization
    x = np.array ([0, 0])
    time = np.vectorize (LN)
    t = time (x, b, k)
    x = q * (t == np.min (t))
    print (t, x)
    # update
    out = []
    for i in range (3):
        print ('---%d---'%(i+1))
        t, y, x, z, alpha, conv = AN_UE_LN (x)
        t = np.round (t, 1)
        x = np.round (x, 2)
        out.append ([i + 1, t, y, x, round (z, 2), round (alpha, 4), round (conv, 4)])
    out = pd.DataFrame (out, columns=['step', 'time', 'auxiliary', 'flow', 'objective', 'alpha', 'convergence'])
    print (out)


def test_SO_LN():
    # parameter
    q = 8
    b = np.array ([4, 8])
    k = np.array ([8, 2])

    # initialization
    x = np.array ([0, 0])
    time = np.vectorize (LN)
    t = time (x, b, k)
    x = q * (t == np.min (t))
    print (t, x)
    # update
    out = []
    for i in range (10):
        t, y, x, z, alpha, conv = AN_SO_LN (x)
        t = np.round (t, 1)
        x = np.round (x, 2)
        out.append ([i + 1, t, y, x, round (z, 2), round (alpha, 4), round (conv, 4)])
    out = pd.DataFrame (out, columns=['step', 'time', 'auxiliary', 'flow', 'objective', 'alpha', 'convergence'])
    print (out)


if __name__ == '__main__':
    test_UE_LN ()
