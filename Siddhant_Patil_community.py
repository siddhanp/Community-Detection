import os
import sys
import itertools
import copy
import time
import networkx as nx

os.environ['SPARK_HOME']="/Users/USC/Desktop/spark-1.6.1-bin-hadoop2.4"
sys.path.append("/Users/USC/Desktop/spark-1.6.1-bin-hadoop2.4/python")

from pyspark import SparkContext
sc = SparkContext()

start_time = time.time()

# TAKE INPUT
path = os.path.join(sys.argv[1])
data = sc.textFile(path).map(lambda x: [x.split(",")[0], x.split(",")[1], x.split(",")[2]]).collect()
edges = []

def buildGraph(data, edges):
    
    data_dict= {}

    for i in range(len(data)):
        user, movie = int(data[i][0]), int(data[i][1])
        data[i][0], data[i][1] = int(data[i][0]), int(data[i][1])
        if user not in data_dict:
            data_dict[user] = set()
        data_dict[user].add(movie)

    for i in itertools.combinations(data_dict.keys(), 2):
        if len(data_dict[i[0]].intersection(data_dict[i[1]])) >= 3:
            edges.append(tuple(sorted(i)))

    graph = nx.Graph()
    #edges = [('A','B'), ('A','C'), ('C', 'B'), ('B','D'), ('D','E'), ('E','F'), ('F','G'), ('G','D'), ('D','F')]
    #edges = [(1,2), (1,3), (3, 2), (2,4), (4,5), (5,6), (6,7), (7,4), (4,6)]
    
    graph.add_edges_from(edges)
    return graph, edges

def bfs_pred(g, source):
    
    bfs, pred, discovered = [], {}, {}
    discovered[source], Q = 0, [source]
    sigma = dict.fromkeys(g, 0.0)
    sigma[source] = 1.0
    for node in g:
        pred[node] = []
    
    while Q:   
        v = Q.pop(0) 
        c, sigmav = discovered[v], sigma[v]
        bfs.append(v)
        for child in g[v]:
            
            if child not in discovered:
                Q.append(child)
                discovered[child] = c + 1
                
            if discovered[child] is c + 1: 
                pred[child].append(v) 
                sigma[child] += sigmav
                               
    return bfs, pred, sigma


def credit_calculation(bfs, pred, sigma, source, b_values):
    
    edge_contribution = dict.fromkeys(bfs, 0)

    while bfs:
        node = bfs.pop()
        w = (1.0 + edge_contribution[node]) / sigma[node]

        for parent in pred[node]:
            c, t = sigma[parent] * w, (parent, node)
            if t in b_values:
                b_values[t] += c
            else:
                b_values[tuple(reversed(t))] += c               
            edge_contribution[parent] += c

    return b_values


def compute_modularity(g2, prev, b_values):

    m, s, degrees = nx.number_of_edges(g2), 0, {}
    coms = list(nx.connected_components(g2))
    prev = len(coms)
    communities = nx.connected_components(g2)

    for node in g2.nodes():
        degrees[node] = g2.degree(node)
        
    #print "Number of Communities", prev
    for c in communities:
        for pair in itertools.product(c, repeat = 2):
            k1, k2 = degrees[pair[0]], degrees[pair[1]]
            #if g2.has_edge(pair[0], pair[1]) or g2.has_edge(pair[1], pair[0]):
            if (pair[0], pair[1]) in b_values or (pair[1], pair[0]) in b_values:
                a = 1
            else:
                a = 0
                
            if m is 0:
                s += 0
            else:
                s += (a - ((0.5*k1*k2)/m))
            #print pair, (a - ((0.5*k1*k2)/m))

    if m is 0:
        return 0, coms, prev
    
    return (0.5*s)/m, coms, prev


# BUILD GRAPH
data = data[1:]
g, edges = buildGraph(data, edges)


# CALCULATE BETWEENNESS
b_values = dict.fromkeys(g.edges(), 0.0)
for source in g:
    bfs, pred, sigma = bfs_pred(g, source)
    b_values = credit_calculation(bfs, pred, sigma, source, b_values)
    
b_values.update((k, v / 2.0) for k, v in b_values.items())
b_values.update((k, int(v * 10) / 10.0) for k, v in b_values.items())
#b_values.update((k, round(v, 1)) for k, v in b_values.items())

new_dict= {}
for k, v in b_values.items():
    if v not in new_dict:
        new_dict[v] = []
    new_dict[v].append(k)

order = sorted(new_dict.iterkeys(), reverse = True)
#print order, new_dict, b_values

# OUTPUT BETWEENNESS VALUES
output_betweenness = []
for i in b_values.iterkeys():
    output_betweenness.append((i[0], i[1], b_values[i]))
output_betweenness = sorted(output_betweenness, key = lambda element: (element[0], element[1]))

myfile = open(sys.argv[3], 'w')
for i in output_betweenness:
    myfile.write("("+str(i[0])+","+str(i[1])+","+str(i[2])+")")
    myfile.write("\n")
myfile.close()
    
#print "TIME FOR BETWEENNESS", str(time.time() - start_time)


# CALCULATE MODULARITY & FIND COMMUNITIES
g2, prev, i, pipe_mod, pipe_com = copy.deepcopy(g), 1, 0, [], []
while len(new_dict) > 0:

    to_be_removed = new_dict[order[i]]
    new_dict.pop(order[i])

    for edge in to_be_removed:
        #print order[i], edge
        if g2.has_edge(edge[0], edge[1]):
            g2.remove_edge(edge[0], edge[1])
        else:
            g2.remove_edge(edge[1], edge[0])

        if (edge[0], edge[1]) in b_values:
            b_values.pop((edge[0], edge[1]))
        else:
            b_values.pop((edge[1], edge[0]))
    
    i += 1
    modularity, communities, prev = compute_modularity(g2, prev, b_values)
    #print modularity
            
    pipe_mod.append(modularity)
    pipe_com.append(communities)

            
# OUTPUT COMMUNITIES
best = pipe_com[pipe_mod.index(max(pipe_mod))]
myfile = open(sys.argv[2], 'w')
for i in best:
    myfile.write(str(sorted(i)))
    myfile.write("\n")
myfile.close()

    
#print "The total execution time taken is " +str(time.time() - start_time)+ " sec."



