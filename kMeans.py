#Aidan Martin 
import numpy as np
import math
import random
from functools import reduce

def update_assignments(instances, centers):
    #move centroids/assign points to closest
    n = len(instances)
    cluster_ids = n*[0]
    for i in range(n):
        distances = [math.dist(instances[i],centers[j]) for j in range(len(centers))]
        cluster_ids[i] = distances.index(min(distances))
    return cluster_ids

def compute_new_means(instances, cid, centers):
    K = len(centers)
    n = len(cid)
    for i in range(K):
        one_cluster = [j for j in range(n) if cid[j] == i]
        cluster_size = len(one_cluster)
        if cluster_size == 0:  # empty cluster
            raise Exception("kmeans: empty cluster created.")
        sum_cluster = reduce(lambda x, y: [p+q for p,q in zip(x,y)], [instances[j] for j in one_cluster])
        centers[i] = [x/cluster_size for x in sum_cluster]
    c_sse = [0 for i in range(K)]
    for point, label in zip(items, cid):
        c_sse[label] += np.square(np.array(point) - np.array(centers[label])).sum()
    return sum(c_sse)/len(c_sse)

def kMeans(items, k, max_iter, min_diff):
    scores=[]
    for r in range(10): #run with 10 random centroid initializations
        iter=0
        SSE=0
        prev=1
        centers = random.sample(items, k)
        cid = update_assignments(items, centers)
        while(True):
            if(iter>max_iter):
                break
            if(abs(prev-SSE)<min_diff):
                break
            prev=SSE
            cid = update_assignments(items, centers)
            SSE=compute_new_means(items, cid, centers)
            iter+=1
        scores.append(SSE)
    return np.average(np.array(scores), axis=0)

#read input data
file = open('data/data2/seeds.txt',mode='r')
seeds = file.read().split('\n')
file.close()
items = []
for line in seeds:
    features = []
    for sample in line.split():
        features.append(float(sample))
    items.append(features)

k=[3,5,7]
for val in k:
    print("K="+str(val)+" Average SSE over 10-runs: "+str(kMeans(items,val,100,0.001)))
