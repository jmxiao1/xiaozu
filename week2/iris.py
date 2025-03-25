import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import pairplot
from kmeans import KMeans

data = pd.read_csv('iris.csv')#读取数据
print(data)#打印数据
iris_types= ['setosa', 'versicolor', 'virginica']#显示种类

x_axis='Sepal.Length'#选取两个特征
y_axis='Sepal.Width'
#先画个图看看按种类划分的效果
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['Species']==iris_type],data[y_axis][data['Species']==iris_type],label=iris_type)
plt.title('label known')
plt.legend()
plt.subplot(1,2,2)
plt.scatter(data[x_axis][:],data[y_axis][:])#这个是全部的数据（无分类）
plt.title('label unknown')
plt.show()

sns.pairplot(data, hue='Species')
plt.show()#看看特征相关性

#处理数据
num_examples = data.shape[0]
x_train=data[[x_axis,y_axis]].values.reshape(num_examples,2)
#指定好训练所需的参数
k=3#分三个簇
max_iteritions=100#迭代100次

k_means=KMeans(x_train,k)#传数据，开始处理
centroids,closest_centroids_ids=k_means.train(max_iteritions)#返回数据

#对比结果
plt.figure(figsize=(12,5))#第一个和之前一样
plt.subplot(1,2,1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['Species']==iris_type],data[y_axis][data['Species']==iris_type],label=iris_type)
plt.title('label known')
plt.legend()
plt.subplot(1,2,2)
for centroid_id ,centroid in enumerate(centroids):
   current_examples_index= (closest_centroids_ids==centroid_id).flatten()
   plt.scatter(data[x_axis][current_examples_index],data[y_axis][current_examples_index],label=centroid_id)
for centroid_id,centroid in enumerate(centroids):
   plt.scatter(centroid[0],centroid[1],c='black',marker="x")
plt.title('label kmeans')
plt.legend()
plt.show()



