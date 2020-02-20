# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
   dm: demand
   q: queue
-------------------------------------------------

"""
import queue

import numpy as np
from scipy.sparse import csr_matrix
from sympy import *

from ShortestPath import Dijkstra
from ShortestPath import Graph

def TransformIn(ml):
    ml=[[item[0]-1,item[1]-1]+item[2:] for item in ml]
    return ml

def transform(mat):
    mat = np.array (mat)
    mat[mat == float ('inf')] = 0
    mat_dns = csr_matrix (mat)
    return mat_dns

# def VOT_PDF(v):
#     if (v >= 0.2) & (v <= 0.4):
#         f = 12.5 * v - 2.5
#     elif (v > 0.4) & (v <= 0.6):
#         f = -12.5 * v + 7.5
#     else:
#         f = 0
#     return f

def VOT_PPF(v):
    if v < 0.2:
        F = 0
    elif (v >= 0.2) & (v <= 0.4):
        F = 0.5 * pow (5 * v - 1, 2)
    elif (v > 0.4) & (v <= 0.6):
        F = 1 - 0.5 * pow (5 * v - 3, 2)
    else:
        F = 1
    return F


def SPMoneyTime(g, vot):
    '''
    shortest path
    :param g: graph, weight of money and time
    :param vot: value of time
    :return: [shortest_path_money,shortest_path_time]
    '''
    money = np.array (g.weight[0])
    time = np.array (g.weight[1])
    cost = money / vot + time
    print('---%f--'%vot)
    print(transform(cost))
    cost = cost.tolist ()
    gc, path = Dijkstra (cost)
    gc = round (gc[3], 2)  # general cost
    path = path[3]
    d_m = 0
    d_t = 0
    for i in range (len (path) - 1):
        u = path[i]
        v = path[i + 1]
        d_m += money[u, v]
        d_t += time[u, v]
    return [d_m, d_t, gc, path]


def SoluPair(g, q, vot_pair):
    solu_pair = []
    for vot in vot_pair:
        solu_pair.append (SPMoneyTime (g, vot))
    q.put (solu_pair)


def ParaMtd(g, q, maxminpair):
    '''
    parametric method
    '''
    qs = []  # visited q
    SoluPair (g, q, maxminpair)
    while q.qsize ():
        p = q.get ()
        vot = (p[0][0] - p[1][0]) / (p[1][1] - p[0][1])
        spmt = SPMoneyTime (g, vot)
        p_compare = [x[:2] for x in p]
        if spmt[:2] not in p_compare:
            q.put ([p[0], spmt])
            q.put ([spmt, p[1]])
        else:
            qs.append (p[0])
            qs.append (p[1])

    # money decrease
    def TakeMoney(item):
        return item[0]
    qs.sort (key=TakeMoney, reverse=True)
    return qs


def LoadPath(qs, dm, vmin, vmax):
    n = len (qs)
    vs = [vmin]
    xs = []
    for i in range (n - 1):
        vot = (qs[i][0] - qs[i + 1][0]) / (qs[i + 1][1] - qs[i][1])
        vs.append (vot)
    vs.append (vmax)
    for i in range (n):
        xs.append (dm * (VOT_PPF (vs[i + 1]) - VOT_PPF (vs[i])))
    return xs

def test_BI_UE():
    # Edges = [[0, 1, 1, 4], [0, 2, 4, 2],
    #          [1, 2, 1, 3], [1, 3, 3, 5],
    #          [2, 1, 1, 1], [2, 3, 1, 4]]
    Edges = TransformIn([[1,3,4,2], [1,2,1,4],
             [2,3,1,3], [3,2,1,1],
             [3,4,1,4], [2,4,3,5]])
    g = Graph ()
    g.add_nodes (range (4))
    g.add_edges (Edges)

    q = queue.Queue ()  # FIFO queue
    vmin, vmax = [0.2, 0.6]
    dm = 8
    qs = ParaMtd (g, q, [vmax, vmin])
    xs = LoadPath (qs, dm, vmin, vmax)
    print(qs)
    print(xs)

if __name__ == '__main__':
    test_BI_UE ()
