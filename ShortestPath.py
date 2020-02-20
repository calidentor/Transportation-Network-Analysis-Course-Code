# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
-------------------------------------------------

"""

from scipy.sparse import csr_matrix
import numpy as np

def transform(mat):
    mat=np.array(mat)
    mat[mat==float('inf')]=0
    mat_dns = csr_matrix (mat)
    return mat_dns

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


def BellsmanFord(w, r):
    n = len (w)
    dist = {}
    prev = {}
    for i in range (n):
        if i == r:
            dist[i] = 0
        else:
            dist[i] = float ('inf')
            prev[i] = 0
    # for i in range(20):
    for i in range (n - 1):  # 最多N-1次
        change = False
        print ('----%d-----' % i)
        u_visit = []
        u_set = [r]
        while u_set:  # 遍历u，N次
            u = u_set.pop (0)
            # for v in range (1, n):  # 遍历v，N-1次,忽略r
            for v in range (n):  # 遍历v时包括起点
                if w[u][v] != float ('inf'):
                    if (v not in u_visit) and (v not in u_set):
                        u_set.append (v)
                    if w[u][v] + dist[u] < dist[v]:
                        change = True
                        # print ('%d-%d' % (u, v))
                        dist[v] = dist[u] + w[u][v]
                        prev[v] = u
            u_visit.append (u)
        # 如果在这一轮没有改变最短路径，就不再进行下一轮，表示收敛
        if not change:
            break
        print (dist, prev)
    # 根据收敛情况输出结果
    if change == False:
        print (prev)
        print (dist)
        prev = dict (zip (prev.values (), prev.keys ()))
        order = [r]
        for i in range (len (prev)):
            order.append (prev[order[-1]])
        print (order)
        for i in range (1, len (order)):
            node = order[i]
            print ('%d|%3.0f| %s' %
                   (i, dist[i],
                    '->'.join ([str (j) for j in order[:order.index (i) + 1]])))
    else:
        print ('存在负权环可达源点，无解')
    return 1 - change, dist, prev


class Graph (object):
    def __init__(self, *args, **kwargs):
        self.adj = {}
        self.grey = {}
        self.color = {}
        self.black = {}
        self.prev = {}
        self.cycle = []
        self.weight = []
        self.N = 0

    def add_node(self, node):
        if not node in self.nodes ():
            self.adj[node] = []
            # DFS
            self.color[node] = 0
            self.grey[node] = 0
            self.black[node] = 0
            self.prev[node] = -1

    def add_nodes(self, node_list):
        self.N = len (node_list)
        for node in node_list:
            self.add_node (node)

    def nodes(self):
        return self.adj.keys ()

    def initial_weight(self, k, fill):
        self.weight = [[[fill] * self.N
                        for i in range (self.N)]
                       for i in range (k)]

    def add_edge(self, edge):
        u, v, *ws = edge
        if v not in self.adj[u]:
            self.adj[u].append (v)
            for i in range (len (ws)):
                self.weight[i][u][v] = ws[i]

    def add_edges(self, edge_list, fill=float ('inf')):
        k = len (edge_list[0]) - 2
        self.initial_weight (k, fill)
        for edge in edge_list:
            self.add_edge (edge)

    def alter_adj(self, adj_mat):
        for i in range (self.N):
            self.adj[i] = []
        if type (adj_mat).__name__ != 'list':
            adj_mat = adj_mat.tolist ()
        for u in range (self.N):
            for v in range (self.N):
                if adj_mat[u][v]:
                    self.adj[u].append (v)

    def add_weight(self, weight_mat):
        if type (weight_mat).__name__ != 'list':
            weight_mat = weight_mat.tolist ()
        self.weight.append (weight_mat)

    def neg_cycle(self):
        self.CycleJohnson ()
        neg_cycle = []
        k = 0
        for weight in self.weight:
            neg_cycle.append ([])
            for cycle in self.cycle:
                cycle_len = 0
                for i in range (len (cycle) - 1):
                    cycle_len = cycle_len + self.weight[k][cycle[i]][cycle[i + 1]]
                if cycle_len < 0:
                    neg_cycle[k].append (cycle)
            k = k + 1
        return neg_cycle  # 输出负权环列表

    def CycleJohnson(self):
        '''
        直接调用python现成包networkx的simple_cycles功能
        :return:
        '''
        import networkx as nx

        Edges_tuple = []
        for u, vs in self.adj.items ():
            for v in vs:
                Edges_tuple.append ((u, v))
        G = nx.DiGraph (Edges_tuple)
        cycles = list (nx.simple_cycles (G))
        cycles_alter = []
        for cycle in cycles:
            cycles_alter.append (cycle + [cycle[0]])
        self.cycle = cycles_alter

    def DFSCycle(self):
        '''
        深度优先DFS搜索找环
        图中的边只可能是树边或反向边，一旦发现反向边，则表明存在环
        算法的复杂度为O(V)
        但不全，与搜索时的起始点有关
        '''
        global time
        time = 0

        def DFSVisit(u):
            global time
            time = time + 1
            self.grey[u] = time
            self.color[u] = 1
            for v in self.adj[u]:
                if self.color[v] == 0:
                    self.prev[v] = u
                    DFSVisit (v)
                elif self.color[v] == 1:  # 如果碰到的点是灰色，说明有环
                    cycle = [v]
                    x = u
                    for i in range (len (self.prev)):
                        cycle.append (x)
                        if (x == v) | (x == -1):
                            break
                        x = self.prev[x]
                    self.cycle.append (cycle[::-1])
            self.color[u] = 2
            time = time + 1
            self.black[u] = time

        for u in self.nodes ():
            if self.color[u] == 0:
                DFSVisit (u)

    def CheckNegCycleBMF(self):
        '''
        用 Bellsman Ford 检测有无负权环
        :return: bool
        '''
        w = self.weight[0]
        dist = {}
        prev = {}
        r = 0
        for i in range (self.N):
            if i == r:
                dist[i] = 0
            else:
                dist[i] = float ('inf')
                prev[i] = 0
        for i in range (self.N - 1):  # 最多N-1次
            change = False
            u_visit = []
            u_set = [r]
            while u_set:  # 遍历u，N次
                u = u_set.pop (0)
                for v in range (self.N):  # 遍历v时包括起点
                    if w[u][v] != float ('inf'):
                        if (v not in u_visit) and (v not in u_set):
                            u_set.append (v)
                        if w[u][v] + dist[u] < dist[v]:
                            change = True
                            dist[v] = dist[u] + w[u][v]
                            prev[v] = u
                u_visit.append (u)
            # 如果在这一轮没有改变最短路径，就不再进行下一轮，表示收敛
            if not change:
                break
        return bool (1 - change)


def MinCostFlow(g, max_irr=10,silent=False):
    import numpy as np
    w = np.array (g.weight)
    w[w == float ('inf')] = 0
    c, p, x = w
    rp = (p - x) + (x.T)  # residual cost
    rc = c + (-c.T)  # residual capacity
    rc[rp == 0] = 0
    Edges_new = (rp != 0)
    g.alter_adj (Edges_new)
    g.add_weight (rc)
    g.add_weight (rp)
    if not silent:
        print (g.adj)
        print (g.weight[3])
        print (g.weight[4])
    x = []
    for k in range (max_irr):
        nwcs = g.neg_cycle ()[3]
        dt = float ('inf')
        cc = []
        if nwcs:
            for nwc in nwcs:
                delta = float ('inf')
                for i in range (len (nwc) - 1):
                    rp_now = rp[nwc[i], nwc[i + 1]]
                    if rp_now < delta:
                        delta = rp_now
                if delta < dt:
                    dt = delta
                    cc = nwc
            for i in range (len (cc) - 1):
                u, v = cc[i], cc[i + 1]
                rp[u, v] = rp[u, v] - dt
                rp[v, u] = rp[v, u] + dt
            rc = c + (-c.T)
            rc[rp == 0] = 0
            Edges_new = (rp != 0)

            g.alter_adj (Edges_new)
            g.weight[3] = rc.tolist ()
            g.weight[4] = rp.tolist ()
            if not silent:
                print ('-----------%d------------' % k)
                print (g.adj, dt, cc)
                print (g.weight[3])
                print (g.weight[4])
        else:
            if not silent:
                print ('-----------Done!-----------')
            x = rp
            x[rc >= 0] = 0
            x = x.T
            c_sum = sum (sum (x * c))
            break
    return x, c_sum


def LggMultiFlow(gs,max_irr=5):
    import numpy as np
    import copy
    M=int((len(gs.weight)-1)/2) # Number of Commodities
    cs=np.array(gs.weight[:M]) # Cost
    p=np.array(gs.weight[M]) # Capacity
    xs=np.array(gs.weight[-M:]) # Initial Flow
    w=np.zeros((gs.N,gs.N)) # Lagrangian multiplier
    th=1
    for k in range(max_irr):
        print('=================%d================='%k)
        cs=cs+w
        for m in range(M):
            g=copy.deepcopy(gs)
            g.weight=[cs[m].tolist(),p.tolist(),xs[m].tolist()]
            xs[m],_=MinCostFlow(g,silent=True)
        k=k+1
        th=1/k
        x_all=np.sum(xs,0)
        z=x_all-p
        z[x_all==0]=0
        l=np.sum(xs*cs,0)-w*p
        w=w + th * (z)
        w[w<0]=0
        print(l)
        print(z)
        print(w)
        print(xs)



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
    for k in range (max_irr):
        # while(abs(z-l)<err):
        cwt = [x[:2] + [x[2] + w * x[3]] for x in Edges]
        cwt_out = [[x[0] + 1] + [x[1] + 1] + [round (x[2], 3)] for x in cwt]
        print ('------------%d------------' % k)
        print (cwt_out)
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
    return out


def TestLagragian():
    Edges = [[0, 2, 4, 2], [2, 3, 1, 4],
             [0, 1, 1, 2], [1, 2, 1, 3],
             [1, 3, 3, 5]]
    r = 0
    s = 3
    b = 7
    out = Lagragian (Edges, b, r, s)
    print ('------------------------')
    print (out)


def TestDijkstra():
    Edges = [[0, 2, 4], [2, 3, 1],
             [0, 1, 1], [1, 2, 1],
             [1, 3, 3]]
    cwt = [x[:2] + [x[2] + w * x[3]] for edge in Edges]
    # Edges = [[0, 1, 10], [1, 0, 10],
    #          [0, 4, 100], [4, 0, 100],
    #          [1, 2, 50], [2, 1, 50],
    #          [4, 3, 60], [3, 4, 60],
    #          [2, 3, 20], [3, 2, 20],
    #          [0, 3, 30], [3, 0, 30],
    #          [2, 4, 10], [4, 2, 10]]
    Edges2 = [[0, 1, 10], [0, 2, 5],
              [1, 3, 1], [1, 2, 2],
              [2, 1, 3], [2, 3, 9],
              [2, 4, 2], [3, 4, 4],
              [4, 3, 6], [0, 4, 7]]
    Nodes = set ([edge[0] for edge in Edges] +
                 [edge[1] for edge in Edges])
    N = len (Nodes)
    w = SparseWeight (N, Edges)
    print (w)
    s = 0
    dist, path = Dijkstra (w, s)
    print (dist)
    print (path)
    for i in set (path):
        if i != s:
            print ('%d到%d的最短距离为:%2.0f，路径为%s' % (s, i, dist[i], '->'.join ([str (j) for j in path[i]])))


def TestBellsmanFord():
    # Edges = [[0, 1, 6], [0, 2, 7],
    #          [1, 3, 5], [3, 1, -2],
    #          [1, 2, 8], [1, 4, -4],
    #          [2, 3, -3], [2, 4, 9],
    #          [4, 3, 7], [4, 0, 2]]
    # Edges = [[2, 1, 0], [5, 1, -1], # 此算例出bug
    #           [5, 2, 1], [1, 3, 5],
    #           [1, 4, 4], [3, 4, -1],
    #           [3, 5, -3], [4, 5, -3],
    #           [0, 1, 0], [0, 2, 0],
    #           [0, 3, 0], [0, 4, 0], [0, 5, 0]]
    # Edges = [[0, 2, 2], [1, 0, -2],
    #          [1, 2, 3], [2, 1, -3],
    #          [3, 1, -3], [2, 3, 2],
    #          [3, 2, -2]]
    Edges = [[0, 2, 2], [2, 0, -2],
             [0, 1, 2], [1, 0, -2],
             [1, 2, 3], [2, 1, -3],
             [1, 3, 3], [3, 1, -3],
             [3, 2, -2]]
    # Edges = [[2, 0, -2], [1, 0, -2],
    #          [0, 1, 2], [3, 2, -2],
    #          [1, 2, 3], [1, 3, 3],
    #          [3, 1, -3]]
    Nodes = set ([edge[0] for edge in Edges] +
                 [edge[1] for edge in Edges])
    N = len (Nodes)

    w = SparseWeight (N, Edges)
    r = 0
    flag, dist, prev = BellsmanFord (w, r)


def TestDFS():
    # Edges = [[0, 1, 1], [0, 3, 1],
    #          [1, 2, 1], [3, 1, 1],
    #          [2, 3, 1], [4, 2, 1],
    #          [4, 5, 1], [5, 5, 1]]
    Edges = [[0, 2, 2], [2, 0, -2],
             [0, 1, 2], [1, 0, -2],
             [1, 2, 3], [2, 1, -3],
             [1, 3, 3], [3, 1, -3],
             [3, 2, -2]]
    g = Graph ()

    g.add_nodes (range (4))
    g.add_edges (Edges, fill=0)
    g.DFSCycle ()
    neg_cycle = g.neg_cycle ()
    print (neg_cycle)


def TestMinCostFlow():
    '''
    u,v,c(cost),p(capacity),x(initial flow)
    :return:
    '''
    # Edges = [[0, 1, 2, 4, 3], [0, 2, 2, 2, 1],
    #          [1, 2, 1, 2, 0], [1, 3, 3, 3, 3],
    #          [2, 3, 1, 5, 1]]
    # Edges = [[0, 2, 2, 2, 0], [0, 1, 2, 3, 3],
    #          [1, 2, 3, 3, 1], [1, 3, 3, 2, 2],
    #          [2, 3, 2, 2, 1]]
    Edges = [[0, 1, 4, 10, 6], [0, 2, 1, 8, 4],
             [2, 1, 2, 5, 0], [1, 3, 6, 2, 0],
             [2, 3, 3, 10, 4], [1, 4, 1, 7, 6],
             [3, 4, 2, 4, 4]]

    g = Graph ()
    g.add_nodes (range (5))
    g.add_edges (Edges)
    x, c_sum = MinCostFlow (g)
    print (transform(x))
    print(c_sum)

    # l=[]
    # for u,vs in g.adj.items():
    #     for v in vs:
    #         l.append((u,v))
    # print(l)


def TestLggMultiFlow():
    '''
    u,v,c1(cost),c2,...,p(capacity),x1(initial flow),x2...
    :return:
    '''
    # Edges = [[0, 1, 2, 1, 3, 0, 1], [0, 2, 1, 4, 2, 2, 0],
    #          [1, 2, 2, 1, 3, 0, 1], [1, 3, 4, 3, 2, 0, 0],
    #          [2, 3, 2, 1, 2, 2, 1]]
    a=float('inf')
    Edges = [[0, 2,5,5,a,10,0], [1,2,1,1,a,0,0],
             [2,3,1,1,10,10,0], [0,4,1,1,5,0,0],
             [1,5,5,5,a,0,20],[3,4,5,5,a,10,0],
             [3,5,1,1,a,0,0]]
    g = Graph ()
    g.add_nodes (range (6))
    g.add_edges (Edges,fill=0)
    LggMultiFlow (g,max_irr=20)


if __name__ == '__main__':
    # TestBellsmanFord ()
    # TestDFS ()
    TestMinCostFlow()
    # TestLggMultiFlow ()

