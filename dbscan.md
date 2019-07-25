# Density Based Spatial Clustering of Applications with Noise

[link](https://medium.com/@elutins/dbscan-what-is-it-when-to-use-it-how-to-use-it-8bd506293818)


[Wiki](https://en.wikipedia.org/wiki/DBSCAN)

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


