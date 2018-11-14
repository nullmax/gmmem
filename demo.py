import numpy as np
from scipy import io

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
print xlist.shape



