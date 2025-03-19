import numpy as np
import matplotlib.pyplot as plt
from datashader import first

def sigmoid(z):
    return 1/(1+np.exp(-z))#就是y_hat
def cost(w,X,y):#损失函数
    w=np.matrix(w)
    X=np.matrix(X)
    y=np.matrix(y)
    # sigmoid(X*w.T)就是h(x),为预测值
    first=np.multiply(y,np.log(sigmoid(X*w.T)))#y==1的情况
    second=np.multiply((1-y),np.log(1-sigmoid(X*w.T)))#y==0的情况
    return np.sum(-first-second)/len(X)#损失函数
learning_rate=0.01# 学习率
epochs=500# 迭代次数
def gradient_descent(X,y,w,b,learning_rate,epochs):
    m=len(y)
    cost_history=[]
    for epoch in range(epochs):#循环epochs次
        y_hat=sigmoid(np.dot(X,w)+b)#点乘，X维度为400*13，w为13*1，得到一个400*1的矩阵，+b为广播
        loss=cost(w,X,y)#损失函数
        cost_history.append(loss)#记录损失,加到末尾
        #loss:400*1,X:400*13,dw:13*1,所以要转置
        dw=(1/m)*np.dot(X.T,loss) # dw=w-1/m*sum((y_hat-y)*X)
        db=(1/m)*np.sum(loss)
        w=w-learning_rate*dw#更新w和b
        b=b-learning_rate*db
    return w,b,cost_history
w=np.zeros(X_train_norm.shape[1])
w,b,cost_history=gradient_descent(X_train_norm,y_train,w,b,learning_rate,epochs)