## 逻辑回归
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
# 设置中文可以显示
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
# 解决x轴负号不正常显示
mpl.rcParams['axes.unicode_minus']=False

X=[];Y=[]
# 加载数据
def loadDataSet():
    f=open('testSet.txt')
    # 逐行读入数据  使用strip去掉头尾的空格  split根据空格分组
    for line in f.readlines():
        nline=line.strip().split()
        # x 需要添加一列
        X.append([float(nline[0]),float(nline[1])])
        Y.append(int(nline[2]))
    return mat(X).T,mat(Y)

# 定义sigmoid函数
def sigmoid(X):
    return 1.0/(1+exp(-X))

# 逻辑回归实现
def Logistic(X,Y,W,b,M,alpha,Count):
    J=zeros((Count,1))
    for i in range(Count):
        # step1 前向传播
        Z=np.dot(W,X)+b
        A=sigmoid(Z)
        # 计算代价函数
        J[i]=-1/M*(np.dot(Y,np.log(A.T))+np.dot((1-Y),np.log((1-A).T)))
        # step2 反向传播
        dZ=A-Y
        dW=1/M*np.dot(dZ,X.T)
        db=1/M*np.sum(dZ)
        # step3 梯度下降
        W=W-alpha*dW
        b=b-alpha*db
    return A,W,b,J


# 绘制图片
def plotBestFit(X,Y,J,W,M,A):
    # 绘制代价函数曲线
    fig1 = plt.figure(1)
    plt.plot(J)
    plt.title(u'代价函数随迭代次数的变化')
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'代价函数的值')
    # 预测值和实际值的对比
    fig2=plt.figure(2)
    plt.scatter(range(0,M),(Y.T).tolist(),c='b',marker='o')
    plt.scatter(range(0,M),np.rint(A.T).tolist(),c='r',marker='s')
    plt.title(u'预测值和实际值的对比')
    plt.legend(('实际值','预测值'))
    # 绘制最终分类图片
    # 根据训练样本标记不同，分为两类不同的点
    xcord1=[]; ycord1=[]
    xcord2=[]; ycord2=[]
    for i in range(M):
        if int(Y[0,i])==1:
            xcord1.append(X[0,i])
            ycord1.append(X[1,i])
        else:
            xcord2.append(X[0,i])
            ycord2.append(X[1,i])
    fig3 = plt.figure(3)
    plt.scatter(xcord1,ycord1,c='b',marker='o')
    plt.scatter(xcord2,ycord2,c='r',marker='s')
    x=linspace(-3,3,100).reshape(100,1) # 生成一个数组
    print(W)
    y=(-b-W[0,0]*x)/W[0,1]
    plt.plot(x,y,c='y')
    plt.title(u'逻辑分类结果示意图')
    plt.xlabel(u'x')
    plt.ylabel(u'y')
    plt.show()

if __name__ == '__main__':
    X,Y=loadDataSet()
    Nx=X.shape[0]  # 特征数
    M=X.shape[1] # 样本个数
    # 权重和偏置的初始化
    W=np.random.randn(1,Nx)*0.01
    b=0
    # 学习速率
    alpha=0.01
    # 迭代次数
    Count=5000
    A,W,b,J=Logistic(X, Y, W, b, M, alpha, Count)
    plotBestFit(X, Y, J, W, M,A)


