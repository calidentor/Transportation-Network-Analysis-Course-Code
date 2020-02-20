# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :Bellman_Ford: label correcting
-------------------------------------------------
"""

import pandas as pd

def TransformIn(ml):
    ml=[[item[0]-1,item[1]-1]+item[2:] for item in ml]
    return ml
def SparseWeight(Edges):
    Nodes = set ([edge[0] for edge in Edges] +
                 [edge[1] for edge in Edges])
    n = len (Nodes)
    w = [[float ('inf')] * n for i in range (n)]
    for edge in Edges:
        w[int (edge[0])][int (edge[1])] = edge[2]
    return w


def BellmanFord(w, r):
    df_dist = pd.DataFrame ([], columns=range (len (w)))
    n = len (w)
    dist = {}
    path = {}
    k = 0
    for i in range (n):
        if i == r:
            dist[i] = 0
            path[i]=[r]
        else:
            dist[i] = float ('inf')
            path[i] = []
    df_dist = df_dist.append (pd.DataFrame (dist, index=[k]))
    for i in range (n - 1):  # 最多N-1次
        change = False
        print ('----%d-----' % i)
        u_visit = []
        u_set = [r]
        while u_set:  # 遍历u，N次
            k = k + 1
            u = u_set.pop (0)
            for v in range (1, n):  # 遍历v，N-1次,忽略s
                if w[u][v] != float ('inf'):
                    if (v not in u_visit) and (v not in u_set):
                        u_set.append (v)
                    if w[u][v] + dist[u] < dist[v]:
                        change = True
                        print ('%d-%d' % (u, v))
                        dist[v] = dist[u] + w[u][v]
                        path[v] = path[u] + [v]
            df_dist = df_dist.append (pd.DataFrame (dist, index=[k]))
            u_visit.append (u)
        # 如果在这一轮没有改变最短路径，就不再进行下一轮，表示收敛
        if not change:
            break
    if change:
        print ('存在负权环可达源点，无解')
    print (df_dist)
    return 1 - change, dist, path


def TestBellmanFord():
    # Edges = [[0, 1, 6], [0, 2, 7],
    #          [1, 3, 5], [3, 1, -2],
    #          [1, 2, 8], [1, 4, -4],
    #          [2, 3, -3], [2, 4, 9],
    #          [4, 3, 7], [4, 0, 2]]
    # Edges=[[2,1,0],[5,1,-1],
    #         [5,2,1],[1,3,5],
    #         [1,4,4],[3,4,-1],
    #         [3,5,-3],[4,5,-3],
    #         [0,1,0],[0,2,0],
    #         [0,3,0],[0,4,0],[0,5,0]]
    # Edges=[[0,1,6],[0,2,4],
    #        [1,2,6],[1,3,3],
    #        [2,1,1],[2,3,5]]
    # Edges = [[0, 2, 18], [0, 4, 30],
    #          [3, 6, 21], [3, 8, 38],
    #          [4, 7, 22], [5, 8, 20],
    #          [7, 8, 9], [8, 7, 0],
    #          [7, 6, 0], [6, 5, 0],
    #          [5, 4, 0], [4, 3, 0],
    #          [3, 2, 0], [2, 1, 0],
    #          [1, 0, 0]]
    Edges = TransformIn ([[1,2,2], [1,3,8],
             [2,3,5], [3,2,6],
             [2,4,3], [4,3,1],
             [4,5,7], [5,4,4],
             [3,5,0], [5,6,4],
             [4,6,6]])
    w = SparseWeight (Edges)
    r = 0
    flag, dist, path = BellmanFord (w, r)
    for i in set (path):
        if i != r:
            print ('%d到%d的最短距离为:%2.0f，路径为%s' % (r+1, i+1, dist[i], '->'.join ([str (j+1) for j in path[i]])))


if __name__ == '__main__':
    TestBellmanFord ()
