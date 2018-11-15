import math
import copy

import numpy as np
from numpy import random
from numpy import linalg

from scipy import io
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gaussian(x, mu, cov):
    gauss = multivariate_normal(mean=mu, cov=cov)
    return gauss.pdf(x)

def readData():
    dataFile = ['data/1.mat', 'data/2.mat', 'data/3.mat']
    x = []
    y = []
    for file in dataFile:
        dataStruct = io.loadmat(file)
        x.extend(dataStruct['x'])
        y.extend(dataStruct['y'])

    x = np.squeeze(x)
    y = np.squeeze(y)
    xlist = np.array([x,y]).transpose()
    return xlist

def E_step(X, mu, cov, tau):
    N = X.shape[0]
    K = tau.shape[0]
    T = np.zeros((N, K))

    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = gaussian(X, mu[k], cov[k])

    for k in range(K):
        T[:, k] = tau[k] * prob[:, k]
    for i in range(N):
        T[i, :] /= np.sum(T[i, :])
    return T


def M_step(X, T):
    Num, Dimension = X.shape
    K = T.shape[1]

    mu = np.zeros((K, Dimension))
    cov = []
    tau = np.zeros(K)

    for k in range(K):
        SumT = np.sum(T[:, k])
        for d in range(Dimension):
            mu[k, d] = np.sum(np.multiply(T[:, k], X[:, d])) / SumT
        cov_k = np.zeros((Dimension, Dimension))
        for i in range(Num):
            cov_k += T[i, k] * np.dot(np.expand_dims((X[i] - mu[k]),0).T , np.expand_dims((X[i] - mu[k]),0)) / SumT
        cov.append(cov_k)
        tau[k] = SumT / Num
    cov = np.array(cov)
    return mu, cov, tau

def gmm(X, K, EPS):

    _, Dimension = X.shape
    # Initial guess of parameters and initializations
    mu = np.random.rand(K, Dimension)
    cov = np.array([np.eye(Dimension)] * K)
    tau = np.array([1.0 / K] * K)

    # EM loop
    counter=0

    while counter<100:
        counter+=1
        print counter, tau

        old_mu = copy.deepcopy(mu)
        T = E_step(X, mu, cov, tau)
        mu, cov, tau = M_step(X, T)
        if abs(np.linalg.norm(old_mu - mu)) < EPS:
            print('Finish!')
            break
    return mu, cov, tau

# Get data
X_matrix = readData()

K = 2
EPS=1e-8

mu, cov, tau = gmm(X_matrix, K, EPS)
print('mu:',mu)
print('cov',cov)
print('tau',tau)

#print plot in 3D space
fig = plt.figure()
ax = Axes3D(fig)
plotX = X_matrix[:, 0]
plotY = X_matrix[:, 1]

#make the class of sample witg the higher probability
#color different plots
Z = np.zeros((X_matrix.shape[0], K))
for k in range(K):
    Z[:, k] = gaussian(X_matrix,mu[k],cov[k])

plotZ = np.zeros((X_matrix.shape[0],1))
colors = ['r' for i in range(plotZ.shape[0])]
for each in range(Z.shape[0]):
    plotZ[each] = max(Z[each])
    colors[each] = 'r' if np.argmax(Z[each]) == 0 else 'b'

plt.title("GMM")
ax.scatter(plotX, plotY, plotZ, color=colors)
ax.set_xlabel('Frequency')
ax.set_ylabel('Standard Deviation')
ax.set_zlabel('Probability')
plt.show()