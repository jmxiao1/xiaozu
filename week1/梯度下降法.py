import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
data_url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
raw_df=pd.read_csv(data_url,header=None,sep='\s+')#读取数据
complete_data=raw_df.values
house_data=complete_data
feature_names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
feature_nums=len(feature_names)#求样本数量
#house_data=house_data.reshape([house_data.shape[0]//feature_nums,feature_nums])#变矩阵
columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
###CRIM:城镇人均犯罪率
###ZN:25,000平方英尺以上住宅用地比例
###INDUS:城镇非零售商业用地比例
###CHAS:查尔斯河虚拟变量（如果是河流则为1，否则为0）
###NOX:一氧化氮浓度（每千万份）
###RM:每栋住宅的平均房间数
###AGE:1940年以前建造的自有住房比例
###DIS:到波士顿五个就业中心的加权距离
###RAD:径向公路的可达性指数
###TAX:每$10,000的全值财产税率
###PTRATIO:城镇师生比例
###B:计算出的城镇中黑人的比例
###LSTAT:低收入人口（%）
###MEDV:自有住房的中位数价值（$1000）
boston= pd.DataFrame(complete_data,columns=columns)#数据框
print(boston.head())#打印数据框的前五行
boston.info()#打印数据框的简要信息
print(boston.describe())#打印数据框的统计描述
#数据可视化
#散点图
boston.hist(bins=20,figsize=(20,15))
plt.show()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#相关系数矩阵
corr_matrix=boston.corr()
# 创建一个大小为12x10的图形窗口
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.show()
import warnings
warnings.filterwarnings('ignore')
# 绘制散点图，x轴为平均房间数，y轴为房价
plt.scatter(boston['RM'],boston['MEDV'])
plt.xlabel('平均房间数')
plt.ylabel('房价')
plt.title('平均房间数与房价')
plt.show()

# 绘制波士顿房价数据集中LSTAT和MEDV两列数据的散点图
plt.scatter(boston['LSTAT'],boston['MEDV'])
plt.xlabel('低收入人口比例')
plt.ylabel('房价')
plt.title('低收入人口比例与房价')
plt.show()
boston.boxplot(column='RM',by='CHAS')
plt.show()

plt.scatter(boston['TAX'],boston['MEDV'])
plt.xlabel('税率')
plt.ylabel('房价')
plt.title('税率与房价')
plt.show()

plt.scatter(boston['PTRATIO'],boston['MEDV'])
plt.xlabel('教师学生比')
plt.ylabel('房价')
plt.title('教师学生比与房价')
plt.show()

plt.scatter(boston['NOX'],boston['MEDV'])
plt.xlabel('一氧化氮浓度')
plt.ylabel('房价')
plt.title('一氧化氮浓度与房价')
plt.show()
offset=400#前400个样本数据为训练集
X,y=house_data[:,0:13],house_data[:,-1]#X为特征，y为预测值
X_train,y_train=X[:offset],y[:offset]#将样本切开
X_test,y_test=X[offset:],y[offset:]
#归一化值域
def normalize_features(X):
    mu=np.mean(X,axis=0)#求均值
    sigma=np.std(X,axis=0)#求标准差
    return (X-mu)/sigma,mu,sigma
X_train_norm,mu,sigma=normalize_features(X_train)
X_test_norm=(X_test-mu)/sigma
#初始化参数
w=np.zeros(X_train_norm.shape[1])
b=0
####y=w1x+w2x2+...+wmxm+b
#超参数
learning_rate=0.01# 学习率
epochs=500# 迭代次数
#运行梯度下降
def gradient_descent(X,y,w,b,learning_rate,epochs):
    m=len(y)
    cost_history=[]
    for epoch in range(epochs):#循环epochs次
        y_hat=np.dot(X,w)+b#点乘，X维度为400*13，w为13*1，得到一个400*1的矩阵，+b为广播
        loss=y_hat-y#损失函数
        cost=np.sum(loss**2)/(2*m)#均方误差
        cost_history.append(cost)#记录损失,加到末尾
        #loss:400*1,X:400*13,dw:13*1,所以要转置
        dw=(1/m)*np.dot(X.T,loss)#dw=w-1/m*sum((y_hat-y)*X)
        db=(1/m)*np.sum(loss)

        w=w-learning_rate*dw#更新w和b
        b=b-learning_rate*db
    return w,b,cost_history
w,b,cost_history=gradient_descent(X_train_norm,y_train,w,b,learning_rate,epochs)

Min=100
# 遍历cost_history列表中的每个元素
time=0
    # 如果当前元素小于Min，则将当前元素赋值给Min
for i in cost_history:
    if i<Min:
    # 每次循环，time加1
        Min=i
    time+=1
print("最小损失为：",Min)
print("最小损失出现次数为：",time)
y_pred=np.dot(X_test_norm,w)+b
#测试集上计算均方误差
plt.scatter(y_test,y_pred,color="red")
plt.xlabel("实际价钱")
plt.ylabel("预测价格")
plt.title("线性回归预测房价")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="blue",lw=2)
plt.show()
mse=np.mean((y_pred-y_test)**2)#求均方误差
r2=1-mse/np.var(y_test)#求决定系数
rmse=np.sqrt(mse)#求均方根误差
mae=np.mean(np.abs(y_pred-y_test))#求平均绝对误差
print("均方误差为：",mse)
print("均方根误差为：",rmse)
print("平均绝对误差为：",mae)
print("决定系数R2为：",r2)



