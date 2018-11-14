import math

import numpy as np
from numpy import random
from numpy import linalg

from scipy import io
from scipy.optimize import minimize, show_options

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gaussian(x, mu, cov):
    y = x - mu
    X = y.dot(linalg.inv(cov)).dot(y.transpose())
    return 1/math.sqrt(2*math.pi)**len(x)/linalg.det(cov)*math.exp(-X/2)

def normpdf2(x, y, mu, cov):
    z = []
    for i in range(len(x)):
        t=[]
        for j in range(len(y)):
            t.append(gaussian(np.array([x[i][j], y[i][j]]), mu, cov))
        z.append(t)
    return np.array(z)

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

def update_mu(x, p):
    y = []
    for i in range(len(x)):
        y.append(x[i] * p[i])
    return sum(y)/sum(p)

def update_std(x, mu, p):
    y = np.array([[0,0],[0,0]])
    for i in range(len(x)):
        z = x[i] - mu
        z = z.reshape(-1, 1)
        y = y + z.dot(z.transpose()) * p[i]
    return y




def main():
    # Generating data
    sample=readData()

    sample1=sample.transpose()

    plt.figure()
    plt.scatter(sample1[0], sample1[1], marker='o')
    plt.grid(True)
    plt.show()

    # Initial guess of parameters and initializations
    mu1 = np.array([np.mean(sample[0]), np.mean(sample[1])])
    mu2 = np.array([np.mean(sample[2]), np.mean(sample[3])])
    std1 = random.normal(0, len(sample), size=4).reshape(2, 2)
    std2 = random.normal(0, len(sample), size=4).reshape(2, 2)
    pi1 = 0.5
    pi2 = 1-pi1

    # EM loop
    plabel1=np.zeros(len(sample))
    plabel2=np.zeros(len(sample))

    counter=0
    criterion=0.1
    converged=False

    while not converged and counter<100:
        counter+=1

        # Expectation
        # Find the probabilty of labeling data points
        for i in range(len(sample)):
            cdf1=gaussian(sample[i], mu1, std1)
            cdf2=gaussian(sample[i], mu2, std2)

            pi2=1-pi1

            plabel1[i]=cdf1*pi1/(cdf1*pi1+cdf2*pi2)
            plabel2[i]=cdf2*pi2/(cdf1*pi1+cdf2*pi2)

        # Maximization
        # From the labeled data points, 
        # find mean through averaging (aka ML)
        mu1=update_mu(sample, plabel1)
        mu2=update_mu(sample, plabel2)
        std1=update_std(sample, mu1, plabel1)
        std2=update_std(sample, mu2, plabel2)    
        pi1=sum(plabel1)/len(sample)
        pi2=1-pi1

    print mu1
    print mu2
    print std1
    print std2
    print pi1
    print pi2

    # x=np.linspace(sample[0].min(), sample[0].max(), 100)
    # y=np.linspace(sample[1].min(), sample[1].max(), 100)
    # x, y = np.meshgrid(x, y)

    # z = normpdf2(x, y, mu1, std1)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.plot(z, normpdf2(x, y, mu1, std1))
    # plt.plot(z, normpdf2(x, y, mu2, std2))
    # ax.plot_surface(x,y,z, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()

if __name__ == '__main__':
    main()