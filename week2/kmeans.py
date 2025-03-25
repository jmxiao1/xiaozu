import numpy as np
import matplotlib.pyplot as plt

class KMeans:#定义类
    def __init__(self, data,k):
        self.data = data
        self.k = k#簇的数量

#max_iterations是最大迭代次数
    #训练模块
    def train(self, max_iterations):#第一步，先选择k个中心点
        centroids=KMeans.centroids_init(self.data,self.k)#随机找点
       #训练模块，开始训练
        num_examples = self.data.shape[0]
        # 创建一个空的数组，用于存储每个样本最近的质心的索引
        closest_centroids_ids = np.empty((num_examples,1))
        for i in range(max_iterations):
            #得到当前所有样本点到k个中心点的距离
            closest_centroids_ids = KMeans.centroids_find_closest(self.data,centroids)
            #更新得到新的中心点位置
            centroids=KMeans.centroids_compute(self.data,closest_centroids_ids,self.k)
        return centroids,closest_centroids_ids

#选择中心点
    def centroids_init(data, k):
        num_examples=data.shape[0]
        random_ids= np.random.permutation(num_examples)#随机选择id,洗牌
        centroids = data[random_ids[:k],:]#只选择k个，同时把这k个的所有维度都拿过来
        return centroids#返回

#寻找最近的中心点
    def centroids_find_closest(data, centroids):#欧氏距离
        num_examples = data.shape[0]
        k=centroids.shape[0]
        closest_centroids_ids=np.zeros((num_examples,1))#要得到离那个簇最近
        for example_index in range(num_examples):
            distance=np.zeros((k,1))
            for centroids_index in range(k):
                distance_diff=(data[example_index,:]-centroids[centroids_index,:])#计算距离,减去当前的簇，得到差异值
                distance[centroids_index]=np.sum(distance_diff**2)#求和,就是sqrt(x**2+y**2)，但可以不用开方来比较
            closest_centroids_ids[example_index]=np.argmin(distance)#得到最小值，即距离最近的质心
        return closest_centroids_ids
    def centroids_compute(data, closest_centroids_ids, k):#更新中心点参数的函数
        num_features = data.shape[1]#特征个数,就是中心点的维度数
        centroids = np.zeros((k,num_features))#类的个数
        for centroid_id in range(k):
            closest_ids=closest_centroids_ids==centroid_id
            centroids[centroid_id]=np.mean(data[closest_ids.flatten(),:],axis=0)#axis=0表示按列求均值
        return centroids

