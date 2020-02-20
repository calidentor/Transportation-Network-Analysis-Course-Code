# -*- coding: utf-8 -*-
"""
-------------------------------------------------
dijkstra：用字典储存数据
label setting
-------------------------------------------------
"""
import pandas as pd
def TransformIn(ml):
    ml=[[item[0]-1,item[1]-1]+item[2:] for item in ml]
    return ml

def SparseWeight(Edges):
    Nodes = set ([edge[0] for edge in Edges] +
                 [edge[1] for edge in Edges])
    # list二维数组的深浅拷贝
    n = len (Nodes)
    w = [[float ('inf')] * n for i in range (n)]
    for edge in Edges:
        w[int (edge[0])][int (edge[1])] = edge[2]
    return w


def Dijkstra(w, s):
    '''
    :param w: weight，n*n 边权，比如距离
    :param s: start，起始点
    :return:
    '''
    # 临时点的dist
    df_dist = pd.DataFrame ([], columns=range (len (w)))
    temp = {}
    # 永久点的dist
    dist = {}
    # 最短路径path字典
    path = {}
    k = 0
    # 初始化
    for i in range (len (w)):
        if i == s:
            temp[i] = 0
            path[i] = [s]
        else:
            temp[i] = float ('inf')
            path[i] = []
    df_dist = df_dist.append (pd.DataFrame (temp, index=[k]))
    # 循环
    while temp:
        k = k + 1
        u = min (temp, key=temp.get)
        dist[u] = temp.pop (u)
        for v in temp.keys ():
            # v是u的相邻节点，并且距离之和小于原来
            if (w[u][v] != float ('inf')) and \
                    (dist[u] + w[u][v] < temp[v]):
                temp[v] = dist[u] + w[u][v]
                path[v] = path[u] + [v]
        if temp:
            df_dist = df_dist.append (pd.DataFrame (temp, index=[k]))
    print(df_dist)
    return dist, path


def TestDijkstra():
    # Edges = [[0, 1, 10], [1, 0, 10],
    #          [0, 4, 100], [4, 0, 100],
    #          [1, 2, 50], [2, 1, 50],
    #          [4, 3, 60], [3, 4, 60],
    #          [2, 3, 20], [3, 2, 20],
    #          [0, 3, 30], [3, 0, 30],
    #          [2, 4, 10], [4, 2, 10]]
    # Edges=[[0,1,10],[0,2,5],
    #         [1,3,1],[1,2,2],
    #         [2,1,3],[2,3,9],
    #         [2,4,2],[3,4,4],
    #         [4,3,6],[0,4,7]]
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
    s = 0
    dist, path = Dijkstra (w, s)
    for i in set (path):
        if i != s:
            print ('%d到%d的最短距离为:%2.0f，路径为%s' % (s+1, i+1, dist[i], '->'.join ([str (j+1) for j in path[i]])))


if __name__ == '__main__':
    TestDijkstra ()
