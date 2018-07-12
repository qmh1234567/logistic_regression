## 逻辑回归
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
# 设置中文可以显示
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
# 解决x轴负号不正常显示
mpl.rcParams['axes.unicode_minus']=False

Nx=2  # 假设两个特征
M = 100 # 100个样本
Num=5000

X=[];Y=[]
# 加载数据
def loadDataSet():
    f=open('testSet.txt')
    # 逐行读入数据  使用strip去掉头尾的空格  split根据空格分组
    for line in f.readlines():
        nline=line.strip().split()
        # x 需要添加一列
        X.append([1.0,float(nline[0]),float(nline[1])])
        Y.append(int(nline[2]))
    return X,Y

# 定义sigmoid函数
def sigmoid(X):
    return 1.0/(1+exp(-X))

# 梯度下降法
def GardDescent(X,Y,alpha=0.01,num=5000):
    J=zeros((num,1))
    # 只有2个特征
    n = X.shape[0]  # 特征数
    W = ones((n,1))
    m = X.shape[1]  # 样本个数
    for i in range(num):
        Z=np.dot(W.T,X)
        A=sigmoid(Z)
        dW=-np.dot(X,(Y-A).T)  # 根据代价函数对w的偏导得出的
        W=W-alpha*dW  # 迭代更新权重  梯度下降
        J[i]= -1/m*(np.dot(Y, log(A.T))+np.dot((1-Y),log((1 - A).T)))
    return J,A,W

# 绘制图片
def plotBestFit(X,Y,J,A,W):
    # 绘制代价函数曲线
    fig1 = plt.figure(1)
    plt.plot(J)
    plt.title(u'代价函数随迭代次数的变化')
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'代价函数的值')
    # 绘制最终分类图片
    n = Y.shape[1]  # 样本数
    # 根据训练样本标记不同，分为两类不同的点
    xcord1=[]; ycord1=[]
    xcord2=[]; ycord2=[]
    for i in range(n):
        if int(Y[0,i])==1:
            xcord1.append(X[1,i])
            ycord1.append(X[2,i])
        else:
            xcord2.append(X[1,i])
            ycord2.append(X[2,i])
    fig3 = plt.figure(3)
    plt.scatter(xcord1,ycord1,c='b',marker='o')
    plt.scatter(xcord2,ycord2,c='r',marker='s')
    x=linspace(-3,3,100).reshape(100,1) # 生成一个数组
    y=(-W[0,0]-W[1,0]*x)/W[2,0]
    print(shape(x),shape(y))
    plt.plot(x,y,c='y')
    plt.title(u'逻辑分类结果示意图')
    plt.xlabel(u'x')
    plt.ylabel(u'y')
    plt.show()




if __name__ == '__main__':
    X,Y=loadDataSet()
    # 将列表转化为矩阵
    X = mat(X).T
    Y = mat(Y).reshape((100, 1)).T
    print(np.shape(X), np.shape(Y))
    m=Y.shape[1]  # 样本个数
    # 传入梯度下降函数
    J,A,W=GardDescent(X,Y,0.01,5000)
    # 绘图
    plotBestFit(X,Y,J,A,W)



