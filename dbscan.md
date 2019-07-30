# Density Based Spatial Clustering of Applications with Noise

### [link](https://medium.com/@elutins/dbscan-what-is-it-when-to-use-it-how-to-use-it-8bd506293818)


### [Wiki](https://en.wikipedia.org/wiki/DBSCAN)

- data clustering technique. It is a density-based clustering non-parametric algorithm: given a set of points in some space, group together points that are closely packed together (points with many nearby neighbours), marking outliers as points that lie alone in low-density regions (whose nearest neighbors are far away). 

- Consider a set of points in some space to be clustered. Let epsilon be a parameter specifying the radius of a neighborhood with respect to some point. The points are classified as core points (density-)reachable points and outliers:
- A point *p* is a core point if at least `minPts` points are within distance epsilon of it (including p)
- A point *q* is directly reachable from *p* if point *q* is within distance epsilon from core point *p*. 
- A point *q* is reachable from *p* if there is a path *p_1, ... , p_n* with *p_1=p* and *p_n=q*, where each *p_i+1* is directly reachable from *p_i*. This implies all points on the path must be core points, with the possible exception of *q*
- All points not reachable from any other point are outliers or noise points
- If *p* is a core point, then it forms a cluster together with all points (core or non-core) that are reachable from it. Each cluster contains at least one core point; non-core points can be part of a cluster, but they form its "edge" since they cannot be used to reach more points

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/DBSCAN-Illustration.svg/600px-DBSCAN-Illustration.svg.png)
Caption: In the diagram, `minPts = 4`. Point A and the other red points are core points, because the area surrounding these points is in a epsilon radius contain at least 4 points (including itself). Because they are all reachable from one another, they form a single cluster. Points B and C are not core points, but reachable from A (via other core points) and thus belong to the cluster as well. Point N is a noise point that is neither core nor directly reachable. 

- A non-core point (yellow) may be reachable but nothing can be reached from it. 
- Thus, a further notion of connectedness is needed to formally define the extent of the clusters found by DBSCAN. Two points *p* and *q* are density-connected if there is a point *o* such that both *p* and *q* are reachable from *o*. Density-connectedness is symmetric:
1. All points within the cluster are mutually density-connected
2. If a point is density-reachable from any point of the cluster, it is part of the cluster as well

- DBSCAN has two parameters epsilon (eps) and the minimum number of points required to form a dense region (`minPts`). It starts with an arbitrary starting point that has not been visited. This point's epsilon-neighborhood is retrived and if it contains sufficiently many points, a cluster is started. 



### [GforG](https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/)

- Clusters are dense regions separated by regions of lower density points. The key idea is that for each point of a cluster, the neighbourhood of a given radius has to contain a minimum number of points. 

- Why DBSCAN? partitioning methods like k-means and hierarchical clustering work for finding spherical-shaped clusters or convex clusters. They are suitable for compact and well-separated clusters. They are severely affected by the presence of noise and outliers. Real-life data can have clusters of arbitrary shape and contain noise. 

- DBSCAN requires **eps** : defining the neighbourhood around a data point i.e. if the distance is lower or equal to some **eps** then they are considered neighbours. If the `eps` is too small then a large part of the data will be considered outliers. If large then clusters will merge and majority of data points will be in the same clusters. One way to find the best eps value is **k-distance graph**
- **MinPts** : minimum number of neighbors (points) within eps radius. Larger the dataset, the larger value of MinPts must be chosen. As a general rule, minimum `MinPts` can be derived from the number of dimensions D in the dataset as, `MinPts >= D+1`. The minimum value of MinPts must be chosen at least 3. 

![alt text](https://media.geeksforgeeks.org/wp-content/uploads/20190418023034/781ff66c-b380-4a78-af25-80507ed6ff26.jpeg)

- Steps of Algorithm:

1. Find all neighbour points within some eps and identify the core points with more than MinPts neighbours
2. For each core point, if it is not already assigned to a cluster, create a new cluster.
3. Recursively find all its densely connected points and assign them to the same cluster as the core point: 
- A points `a` and `b` are said to be densely connected if there exist a point `c` which has sufficient number of points in its neighbours and both the points `a` and `b` are within the `eps` distance. So if `b` is a neighbour of `c`, `c` is a neighbour of `d`, `d` is a neighbour of `e`, which is a neighbour of `a` this implies `b` is a neighbour of `a`.
4. Iterate through the remaining unvisited points in the dataset. Those points that are not part of a cluster are noise. 

```
DBSCAN(dataset, eps, MinPts){
# cluster index
C = 1
for each unvisited point p in dataset {
         mark p as visited
         # find neighbors
         Neighbors N = find the neighboring points of p

         if |N|>=MinPts:
             N = N U N'
             if p' is not a member of any cluster:
                 add p' to cluster C 
}
```

In python:

```
import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn import metrics 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler 
from sklearn import datasets 
  
# Load data in X 
db = DBSCAN(eps=0.3, min_samples=10).fit(X) 
core_samples_mask = np.zeros_like(db.labels_, dtype=bool) 
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_ 
  
# Number of clusters in labels, ignoring noise if present. 
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
  
print(labels) 
  
# Plot result 
import matplotlib.pyplot as plt 
  
# Black removed and is used for noise instead. 
unique_labels = set(labels) 
colors = ['y', 'b', 'g', 'r'] 
print(colors) 
for k, col in zip(unique_labels, colors): 
    if k == -1: 
        # Black used for noise. 
        col = 'k'
  
    class_member_mask = (labels == k) 
  
    xy = X[class_member_mask & core_samples_mask] 
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                                      markeredgecolor='k',  
                                      markersize=6) 
  
    xy = X[class_member_mask & ~core_samples_mask] 
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, 
                                      markeredgecolor='k', 
                                      markersize=6) 
  
plt.title('number of clusters: %d' %n_clusters_) 
plt.show() 
```

Advantages over k-means:
- K-means forms spherical clusters only, fails when data is not spherical
- K-means is sensitive to outliers, they can skew clusters 
![alt text](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/PicsArt_11-17-08.07.10-300x300.jpg)


Alternative Clustering Methods

- Gaussian Mixtures [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) : distance between point P and distribution D. 

- Spectral Clustering [graph distance](https://en.wikipedia.org/wiki/Spectral_clustering) : uses eignevalues of the similarity matrix to do dimensionality reduction and compares within a lower dimension. 

- Affinity Propagation [graph distance](https://en.wikipedia.org/wiki/Affinity_propagation) : finds "exemplars" that are members of the set that are representative of clusters. 



