import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#我直接从网站读数据，,csdn上得数据集和网站得不同，而且不会转csv
data_url='https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
raw_df=pd.read_csv(data_url,header=None,sep='\s+')#读取数据
complete_data=raw_df.values
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
#根据图里与房价相关性较大的进行绘图，有-0.74的LSTAT,-0.51的PTRATIO,0.7的RM
# 绘制散点图，x轴为平均房间数，y轴为房价
plt.scatter(boston['RM'],boston['MEDV'])
plt.xlabel('平均房间数')
plt.ylabel('房价')
plt.title('平均房间数与房价')
plt.show()

plt.scatter(boston['LSTAT'],boston['MEDV'])
plt.xlabel('低收入人口比例')
plt.ylabel('房价')
plt.title('低收入人口比例与房价')
plt.show()

plt.scatter(boston['PTRATIO'],boston['MEDV'])
plt.xlabel('师生比例')
plt.ylabel('房价')
plt.title('师生比例与房价')
plt.show()

plt.scatter(boston['TAX'],boston['MEDV'])
plt.xlabel('税率')
plt.ylabel('房价')
plt.title('税率与房价')
plt.show()
# 将数据集的特征和标签分开
X=boston.drop('MEDV',axis=1).values
y=boston['MEDV'].values
# 在X中添加一列全为1的列，用于计算截距
#X=np.c_[np.ones(X.shape[0]),X]
# 使用最小二乘法计算回归系数
#theta=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# 将测试集设置为训练集
#X_test=X
# 使用回归系数预测房价
#y_pred=X_test.dot(theta)
# 绘制实际房价和预测房价的散点图
#plt.scatter(y,y_pred)
# 绘制一条直线，表示实际房价和预测房价完全相等
#plt.plot([y.min(),y.max()],[y.min(),y.max()],color='red')
# 设置x轴和y轴的标签
#plt.xlabel('实际房价')
#plt.ylabel('预测房价')
# 设置图表的标题
#plt.title('波士顿房价预测')
# 显示图表
#plt.show()
#mse=np.mean((y_pred-y)**2)#求均方误差
#r2=1-mse/np.var(y)#求决定系数
#rmse=np.sqrt(mse)#求均方根误差
#mae=np.mean(np.abs(y_pred-y))#求平均绝对误差
#print("回归系数为：",theta)
#print("均方误差为：",mse)
#print("均方根误差为：",rmse)
#print("平均绝对误差为：",mae)
#print("决定系数R2为：",r2)
features = boston.columns[:-1]
for feature in features:
    X = boston[[feature]].values#拆分每一个因数进行计算回归系数
    y = boston['MEDV'].values
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)#调用函数进行计算
    X_test = X  # 使用训练数据作为测试数据
    y_pred = X_test.dot(theta)
    #查看在该影响因数下，预测房价和实际房价的变化，画图
    plt.scatter(X_test[:, 1], y)
    plt.plot(X_test[:, 1], y_pred, color='red')
    plt.xlabel(f'影响因素 {feature}')
    plt.ylabel('房价')
    plt.title(f'影响因素 {feature} 对预测房价和实际房价的影响')
    plt.show()
    #观察预测房价和实际房价直接关系
    plt.scatter(y, y_pred)
    plt.plot([y.min(), y.max()], [y_pred.min(), y_pred.max()], color='red')
    plt.xlabel('实际房价')
    plt.ylabel('预测房价')
    plt.title(f'影响因素 {feature} 的波士顿房价预测')
    plt.show()
#求误差
    mse = np.mean((y_pred - y) ** 2)  # 求均方误差
    r2 = 1 - mse / np.var(y)  # 求决定系数
    rmse = np.sqrt(mse)  # 求均方根误差
    mae = np.mean(np.abs(y_pred - y))  # 求平均绝对误差
    print(f"影响因素 {feature} 的回归系数为：", theta)
    print(f"影响因素 {feature} 的均方误差为：", mse)
    print(f"影响因素 {feature} 的均方根误差为：", rmse)
    print(f"影响因素 {feature} 的平均绝对误差为：", mae)
    print(f"影响因素 {feature} 的决定系数R2为：", r2)

#数据是自己跑完后手打回来的，图跑得好像没什么毛病，但是总觉得有点怪（
#影响因素 CRIM 的回归系数为： [24.03310617 -0.41519028]
#影响因素 ZN 的回归系数为： [20.91757912  0.14213999]
#影响因素 INDUS 的回归系数为： [29.75489651 -0.64849005]
#影响因素 CHAS 的回归系数为： [22.09384289  6.34615711]
#影响因素 NOX 的回归系数为： [ 41.34587447 -33.91605501]
#影响因素 RM 的回归系数为： [-34.67062078   9.10210898]
#影响因素 AGE 的回归系数为： [30.97867776 -0.12316272]
#影响因素 DIS 的回归系数为： [18.39008833  1.09161302]
#影响因素 RAD 的回归系数为： [26.38212836 -0.4030954 ]
#影响因素 TAX 的回归系数为： [ 3.29706545e+01 -2.55680995e-02]
#影响因素 PTRATIO 的回归系数为： [62.34462747 -2.1571753 ]
#影响因素 B 的回归系数为： [10.55103414  0.03359306]
#影响因素 LSTAT 的回归系数为： [34.55384088 -0.95004935]
#所以预测房价=-24.03310617-0.41519028*CRIM+20.91757912*ZN+29.75489651*INDUS+22.09384289*CHAS-41.34587447*NOX-34.67062078*RM+30.97867776*AGE+
#           18.39008833*DIS+26.38212836*RAD+3.29706545e+01*TAX+62.34462747*PTRATIO+10.55103414*B+34.55384088*LSTAT

