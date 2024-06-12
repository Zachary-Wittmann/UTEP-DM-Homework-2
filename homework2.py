# Created and formatted by Zachary Wittmann
# Version 1.5

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import numpy as np
import pandas as pd


def RandIndex(result, gt):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, len(result)-1):
        for j in range(i+1, len(result)):
            # TP
            if (result[i] == result[j]):  # Positive
                if (gt[i] == gt[j]):  # True
                    TP += 1
                else:  # False
                    FP += 1
            else:  # Negative
                if (gt[i] == gt[j]):  # False
                    FN += 1
                else:  # True
                    TN += 1
    RI = (TP+TN)/(TP+TN+FP+FN)
    return RI


def Purity(result, gs):
    if (len(result) != len(gs)):
        raise Exception("Two arrays must have the same length.")

    labelList = list(set(result))  # cluster labels
    classList = list(set(gs))  # ground truth labels

    k = len(labelList)
    c = len(classList)
    n = len(result)

    labelDict = dict(zip(labelList, range(k)))
    classDict = dict(zip(classList, range(c)))

    clusterClassDist = np.zeros([k, c])

    for i in range(n):
        clustIndex = labelDict[result[i]]
        classIndex = classDict[gs[i]]
        clusterClassDist[clustIndex][classIndex] += 1

    numerator = 0
    for j in range(k):
        numerator = numerator + max(clusterClassDist[j])

    return numerator/n


def kMeansInfo(data, goldSet, k, printClusters=False):
    # K-means Clustering
    print(f"With k={k}, using k-means clustering:\n")
    km = KMeans(n_clusters=k).fit(data)
    if printClusters:
        print("Cluster Assignments: ", km.labels_)
    for i in range(k):
        clusterCount = (km.labels_ == i).sum()
        print(f"Cluster {i} has {clusterCount} rows")

    print("SSD (closest cluster center (inertia_)): ", km.inertia_)

    rand = RandIndex(km.labels_, goldSet)
    print("Rand index is: ", rand)

    pt = Purity(km.labels_, list(goldSet.flatten()))
    print("Purity: ", pt)

    sc = silhouette_samples(data, km.labels_)
    print("MinSC: ", np.min(sc))
    print("AvgSC: ", np.mean(sc))
    print("MaxSC: ", np.max(sc))


def agglomerativeInfo(data, goldSet, k, printClusters=False):
    print(f"\nAgglomerative clustering with {k} clusters: \n")
    ac = AgglomerativeClustering(n_clusters=k).fit(data)
    if printClusters:
        print("Agglomerative Clustering Results: ", ac.labels_)
    for i in range(k):
        clusterCount = (ac.labels_ == i).sum()
        print(f"Cluster {i} has {clusterCount} rows")

    # Rand Index
    rand = RandIndex(ac.labels_, goldSet)
    print("Agglomerative Clustering Rand Index Results: ", rand)

    # Purity
    pt = Purity(ac.labels_, list(goldSet.flatten()))
    print("Agglomerative Clustering Purity Results: ", pt)

    # ASC
    sc = silhouette_samples(data, ac.labels_)
    print("Agglomerative Clustering MinSC Results: ", np.min(sc))
    print("Agglomerative Clustering AvgSC Results: ", np.mean(sc))
    print("Agglomerative Clustering MaxSC Results: ", np.max(sc))


def dbScanInfo(data, goldSet, k, eps, min_pts, printClusters=False):
    print("\nDBSCAN: \n")
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
    if printClusters:
        print("DBSCAN labels: ", db.labels_)
    # Removes 1 from the length to remove outliers
    clusters = len(set(db.labels_)) - 1
    print((f'With eps={eps} and min_pts={min_pts}, there are '
           f'{clusters} clusters present.\n'))
    outliers = (db.labels_ == -1).sum()
    print((f'With eps={eps} and min_pts={min_pts}, the number'
           f' of outliers in the practice dataset is {outliers}.\n'))


if __name__ == "__main__":

    dataCSV = 'Data_Dry_Beans.csv'
    goldSetCSV = 'Benchmark_Dry_Beans.csv'
    # dataCSV = 'practicedata.csv'
    # goldSetCSV = 'practicegoldset.csv'

    df = pd.read_csv(dataCSV)
    gs = pd.read_csv(goldSetCSV, header=None)

    data = df.values
    goldSet = gs.values
    k = 7
    eps = 500.0
    min_pts = 7

    # Class Labels
    classLabels = len(set(list(gs.values.flatten())))
    print(f"There are {classLabels} Class Labels in {goldSetCSV}\n")

    kMeansInfo(data, goldSet, k)
    agglomerativeInfo(data, goldSet, k)
    dbScanInfo(data, goldSet, k, eps, min_pts)
