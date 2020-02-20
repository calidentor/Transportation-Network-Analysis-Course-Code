# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
-------------------------------------------------

"""

import numpy as np
import pandas as pd
from sympy import *


def LN(x, b, k):
    '''
    link: flow -> time
    LN: linear function <- t=b+kx
    '''
    t = x * k + b
    return t


def bisection(func, iter=5):
    '''
    bisection method
    :param func: func(x)=0
    :param irr: iteration times
    :return: root
    '''
    left = 0
    right = 1
    eps = 10e-3
    mid = (left + right) / 2
    i = 0
    while abs (func (mid)) > eps:
    # while i < iter:
        mid = (left + right) / 2
        # if func (left) * func (mid) <= 0:
        if func (mid) < 0:
            left = mid
        elif func (mid) > 0:
            right = mid
        else:
            print (mid)
            break
        print (mid)
        if i >= iter-1:
            break
        i = i + 1
    return mid


def OBJ_UE_LN(x, b, k):
    '''
    UE objective function z(x)
    z(x)=integrate(t(w),(w,0,x))
    '''
    w = symbols ('w')
    z = integrate (LN (w, b, k), (w, 0, x))
    return z


def AN_UE_LN(x, b, k, q):
    # np.vectorize: make function can run for vector
    time = np.vectorize (LN)
    obj = np.vectorize (OBJ_UE_LN)
    t = time (x, b, k) # time
    z = obj (x, b, k) # objective
    z = sum (z)
    # all or nothing assignment
    y = np.zeros (len (x))
    y[np.argmin (t)] = 1
    y = y * q
    # derivative of z(x): d[z(x)]/dx
    def df_obj(alpha):
        df = (y - x) * time (x + alpha * (y - x), b, k)
        df = sum (df)
        return df
    # d[z(x)]/dx = 0 -> alpha
    alpha = bisection (df_obj)
    x1 = x + alpha * (y - x)
    # convergence
    conv = sqrt (sum (pow (x1 - x, 2))) / sum (x)
    return t, y, x1, z, alpha, conv


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
    for i in range (5):
        print ('---%d---' % (i + 1))
        t, y, x, z, alpha, conv = AN_UE_LN (x, b, k, q)
        t = np.round (t, 1)
        x = np.round (x, 2)
        out.append ([i + 1, t, y, x, round (z, 2), round (alpha, 4), round (conv, 4)])
    out = pd.DataFrame (out, columns=['step', 'time', 'auxiliary', 'flow', 'objective', 'alpha', 'convergence'])
    print (out)


if __name__ == '__main__':
    test_UE_LN ()
