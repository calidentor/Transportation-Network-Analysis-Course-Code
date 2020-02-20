# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description : constrained shortest path problem
-------------------------------------------------

"""

def TransformOut(mat):
    hash=[]
    n=len(mat)
    for i in range(n):
        for j in range(n):
            x=mat[i][j]
            if (x!=0)&(x!=float('inf')):
                hash.append([(i+1,j+1),x])
    return hash

def SparseWeight(N, Edges, padding=float ('inf')):
    '''
    数据结构list，未用numpy
    :param Edges: Edges 节点下标从0开始
    :return: N*N的稀疏矩阵
    '''
    # Nodes = set ([edge[0] for edge in Edges] +
    #              [edge[1] for edge in Edges])
    # N = len (Nodes)
    w = [[padding] * N for i in range (N)]
    for edge in Edges:
        w[int (edge[0])][int (edge[1])] = edge[2]
    return w


def Dijkstra(w, s=0):
    '''
    :param w: weight，n*n 边权，比如距离
    :param s: start，起始点
    :return:
    '''
    # 临时点的dist
    temp = {}
    # 永久点的dist
    dist = {}
    # 最短路径path字典
    path = {}
    # 初始化
    for i in range (len (w)):
        if i == s:
            temp[i] = 0
            path[i] = [s]
        else:
            temp[i] = float ('inf')
            path[i] = []
    # 循环
    while temp:
        u = min (temp, key=temp.get)
        dist[u] = temp.pop (u)
        for v in temp.keys ():
            # v是u的相邻节点，并且距离之和小于原来
            if (w[u][v] != float ('inf')) and \
                    (dist[u] + w[u][v] < temp[v]):
                temp[v] = dist[u] + w[u][v]
                path[v] = path[u] + [v]
    return dist, path


def Lagragian(Edges, b, r, s):
    '''
    :param Edges: [i,j,cost(min),time(costrain)]
    :param b: constrain
    :param r: origin,first node
    :param s: source,last node
    :return:
    '''
    import numpy as np
    import pandas as pd

    Costs = [x[:3] for x in Edges]
    Times = [x[:2] + [x[3]] for x in Edges]
    Nodes = set ([edge[0] for edge in Edges] +
                 [edge[1] for edge in Edges])
    N = len (Nodes)
    c = SparseWeight (N, Costs)
    c0 = SparseWeight (N, Costs, 0)
    t = SparseWeight (N, Times)
    t0 = SparseWeight (N, Times, 0)
    th = 1
    w = 0
    err = 10e-5
    out = []
    max_irr = 10
    cwts=[] # c+w*t list for print
    for k in range (max_irr):
        # while(abs(z-l)<err):
        cwt = [x[:2] + [x[2] + w * x[3]] for x in Edges]
        cwt_out = [[x[0] + 1] + [x[1] + 1] + [round (x[2], 3)] for x in cwt]
        print ('------------%d------------' % k)
        print (cwt_out)
        cwts.append(cwt_out)
        cwt = SparseWeight (N, cwt)
        dist, path = Dijkstra (cwt, r)
        dist = dist[s]
        path = path[s]
        l = dist - b * w
        x = [path[i:i + 2] + [1] for i in range (len (path) - 1)]
        x = SparseWeight (N, x, 0)
        z = np.array (c0) * np.array (x)
        z = sum (sum (z))
        tx = np.array (t0) * np.array (x)
        tx = sum (sum (tx))
        path_out = [str (i + 1) for i in path]
        path_out = '->'.join (path_out)
        out.append ([k, round (w, 4), path_out, round (dist, 3), round (tx, 3), round (l, 3), round (z, 3)])
        if ((k > 0) & (abs (l - z) < err)):
            break
        k = k + 1
        th = 1 / k
        w = w + th * (tx - b)
        w = max (0, w)
    out = pd.DataFrame (out, columns=['k', 'w', 'path', 'dist', 'tx', 'Lw', 'zx'])
    return out,cwts


def TestLagragian():
    '''
    Edges: i,j,cost(min),time(constrained)
    '''
    Edges = [[0, 2, 4, 2], [2, 3, 1, 4],
             [0, 1, 1, 2], [1, 2, 1, 3],
             [1, 3, 3, 5]]
    r = 0
    s = 3
    b = 7
    out,cwts = Lagragian (Edges, b, r, s)
    print ('------------------------')
    print (out)
    print(cwts)

if __name__ == '__main__':
    TestLagragian ()
