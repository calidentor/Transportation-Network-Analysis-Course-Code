# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Description :
-------------------------------------------------

"""

def TransformIn(ml):
    ml=[[item[0]-1,item[1]-1]+item[2:] for item in ml]
    return ml

def TransformOut(mat):
    hash=[]
    n=len(mat)
    for i in range(n):
        for j in range(n):
            x=mat[i][j]
            if (x!=0)&(x!=float('inf')):
                hash.append([(i+1,j+1),x])
    return hash
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

def MinCostFlow(g, max_irr=10,silent=False):
    import numpy as np
    w = np.array (g.weight)
    w[w == float ('inf')] = 0
    c, p, x = w
    rc = c + (-c.T)  # residual cost
    rp = (p - x) + (x.T)  # residual capacity
    rc[rp == 0] = 0
    Edges_new = (rp != 0)
    g.alter_adj (Edges_new)
    g.add_weight (rc)
    g.add_weight (rp)
    if not silent:
        out_c = TransformOut (g.weight[3])
        out_p = TransformOut (g.weight[4])
        out_cp = []
        for i in range (len (out_c)):
            print ([out_c[i][0]] + [(out_c[i][1], out_p[i][1])])
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
                print('所有负权环:')
                for nwc_ in nwcs:
                    print('->'.join ([str (i + 1) for i in nwc_]))
                print ('本次负权环', '->'.join ([str (i + 1) for i in cc]))
                print ('负权环增量:%.1f'%dt)
                out_c = TransformOut (g.weight[3])
                out_p = TransformOut (g.weight[4])
                out_cp = []
                for i in range (len (out_c)):
                    print ([out_c[i][0]] + [(out_c[i][1], out_p[i][1])])
        else:
            if not silent:
                print ('-----------Done!-----------')
            x = rp
            x[rc >= 0] = 0
            x = x.T
            c_sum = sum (sum (x * c))
            break
    return x, c_sum

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
    # Edges = [[0, 1, 4, 10, 6], [0, 2, 1, 8, 4],
    #          [2, 1, 2, 5, 0], [1, 3, 6, 2, 0],
    #          [2, 3, 3, 10, 4], [1, 4, 1, 7, 6],
    #          [3, 4, 2, 4, 4]]
    # Edges = TransformIn(
    #     [[1, 2, 6, 20, 0], [1, 3, 7, 30, 25],
    #      [3, 2, 5, 25, 25], [3, 4, 4, 10, 0],
    #      [2, 4, 2, 20, 0], [2, 5, 2, 25, 25],
    #      [4, 5, 1, 20, 0]])
    Edges = TransformIn(
        [[1, 2, 1,3,2], [1, 3, 4,2,1],
         [2,3,1,3,0], [3,4,2,2,1],
         [2,4,4,2,2]])

    g = Graph ()
    g.add_nodes (range (4))
    g.add_edges (Edges)
    x, c_sum = MinCostFlow (g)
    x_out=TransformOut(x)
    for i in x_out:
        print(i)
    print(c_sum)
if __name__ == '__main__':
    TestMinCostFlow()
