# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description : bi-criterion shortest path problem
-------------------------------------------------
"""

def SparseWeight2(N, Edges):
    '''
    :param N: number of nodes
    :param Edges: [[u,v,c,t],..]
    :return: 2 N*N weight mat for cost and time
    '''
    c = [[float ('inf')] * N for i in range (N)]
    t = [[float ('inf')] * N for i in range (N)]
    for edge in Edges:
        u, v, c_, t_ = edge
        c[u][v] = c_
        t[u][v] = t_
    return c, t


def LabelCorrecting(c, t, r):
    N = len (c)
    dist = {}
    path = {}
    k = 0
    for i in range (N):
        if i == r:
            dist[i] = [[0, 0]]
            path[i] = [r]
        else:
            dist[i] = [float ('inf'), float ('inf')]
            path[i] = []
    for i in range (N - 1):  # 最多N-1次
        change = False
        print ('----%d-----' % i)
        u_visit = []
        u_set = [r]
        while u_set:  # 遍历u，N次
            k = k + 1
            u = u_set.pop (0)
            for v in range (1, N):  # 遍历v，N-1次,忽略s
                if c[u][v] != float ('inf'):
                    if (v not in u_visit) and (v not in u_set):
                        u_set.append (v)
                    c_=c[u][v] + dist[u][0]
                    t_=t[u][v] + dist[u][1]
                    if (c[u][v] + dist[u][0] <= dist[v][0])|(t[u][v] + dist[u][1] <= dist[v][1]):
                        change = True
                        print ('%d-%d' % (u, v))
                        dist[v].append()
                        dist[v][0] = dist[u][0] + c[u][v]
                        dist[v][1] = dist[u][1] + t[u][v]
                        path[v] = path[u] + [v]
            u_visit.append (u)
        # 如果在这一轮没有改变最短路径，就不再进行下一轮，表示收敛
        if not change:
            break
    if change:
        print ('存在负权环可达源点，无解')
    return 1 - change, dist, path


def ShortestPathBiCriterion():
    pass


def TestShortestPathBiCriterion():
    Edges = [[0, 1, 1, 4], [0, 2, 4, 2],
             [1, 2, 1, 3], [1, 3, 3, 5],
             [2, 1, 1, 1], [2, 3, 1, 4]]
    c, t = SparseWeight2 (4, Edges)
    print (c)
    r = 0  # origin node
    flag, dist, path=LabelCorrecting (c, t, r)
    print(dist)
    print(path)


if __name__ == '__main__':
    TestShortestPathBiCriterion ()
