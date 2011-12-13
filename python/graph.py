from heapq import *
def link(x, y):
    return str(x) + ',' + str(y)

def read_matrix(fname):
    rows = open(fname).readlines()
    matrix = []
    for row in rows:
        matrix.append([int(val) for val in row.split(',')])
    return matrix
    
def get_graph(matrix, R=True,D=True,U=True,L=True):
    graph = {}
    num_rows = len(matrix)
    for m in range(num_rows + 1):
        row_len = len(matrix[m - 1])
        u = m - 1
        d = m + 1
        for n in range(row_len + 1):
            cur = link(m, n)
            if (m > 0 and n > 0):
                graph[cur] = [{},float('inf'), 0, []]
            else:
                graph[cur] = [{}, 0, 0, []]
            l = n - 1
            r = n + 1
            if (n <= row_len and n > 0):
                if (u > 0): 
                    graph[cur][0].update({link(u, n): matrix[u-1][n-1]})
                if (d <= row_len):
                    graph[cur][0].update({link(d, n): matrix[d-1][n-1]})
            if (m <= num_rows and m > 0):
                if (l > 0): 
                    graph[cur][0].update({link(m, l): matrix[m-1][l-1]})
                if (r <= num_rows): 
                    graph[cur][0].update({link(m, r): matrix[m-1][r-1]})
    return graph

def dijkstra(graph, h):
    count = 0
    while len(h):
        old_dist, parent = heappop(h)
        for child,weight in graph[parent][0].items():
            if graph[child][2] == 1: continue
            cur_dist = graph[child][1] 
            new_dist = old_dist + weight
            if new_dist < cur_dist:
                graph[child][1] = new_dist
                graph[child][3] = list(graph[parent][3])
                graph[child][3].append([parent, graph[parent][1]])
            if (graph[child][2] == 0):
                graph[child][2] = 1
                heappush(h, (new_dist, child))
        graph[parent][2] = 2
    return graph
