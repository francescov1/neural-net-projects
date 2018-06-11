import numpy as np
from matplotlib import pyplot as plt

def plotData(r1, r2, label1, label2):

    plt.plot(setosa[:,r1], setosa[:,r2], '.g', label='setosa')
    plt.plot(versi[:,r1], versi[:,r2], '.r', label = 'versi')
    plt.plot(virgin[:,r1], virgin[:,r2], '.b', label='virgin')
    plt.legend()
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.axis('auto')
    plt.show()

txtData = np.genfromtxt("train.txt", dtype = float, delimiter = ",")
s = len(txtData[:,0]) # num of samples

data = np.zeros((s, 4))
data = txtData[:, :-1]

setosa = data[:-80, :]
versi = data[40:-40, :]
virgin = data[80:, :]

fig = plt.figure()
#plotData(0, 1, 'Sepal Length (cm)', 'Sepal Width (cm)')
plotData(2, 3, 'Petal Length (cm)', 'Petal Width (cm)')

#plotData(0, 2, 'Sepal Length (cm)', 'Petal Length (cm)')
#plotData(1, 3, 'Sepal Width (cm)', 'Petal Width (cm)')







