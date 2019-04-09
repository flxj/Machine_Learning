# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 计算欧式距离
def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))

# 初始化聚类中心
def initCentroids(data, k):
    n_samples, n_features = data.shape
    centroids = np.zeros((k, n_features))
    for i in range(k):
        index = int(np.random.uniform(0, n_samples))
        centroids[i, :] = data[index, :]
    return centroids

# K-Means聚类
def kmeans(data, k):
    n_samples = data.shape[0]
    # 第一列记录各个样本在本轮训练中所属的类别
    # 第二列记录各个样本距类别中心的距离
    clusterAssment = np.mat(np.zeros((n_samples, 2)))
    clusterChanged = True

    ## 第一步：初始化聚类中心
    centroids = initCentroids(data, k)

    while clusterChanged:
        clusterChanged = False
        ## 依次计算每个样本到k个聚类中心的距离
        for i in range(n_samples):
            minDist  = float('inf')
            minIndex = 0
            ## 最近的聚类中心
            for j in range(k):
                distance = euclDistance(centroids[j, :], data[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j

            ## 更新该样本类别
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2

        ## 更新聚类中心
        for j in range(k):
            pointsInCluster = data[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)

    print('Congratulations, cluster complete!')
    return centroids, clusterAssment

