import numpy as np
from matplotlib import pyplot as plt

class PerceptronNetwork:
    # X = input data, Y = expected output, wi = initial threshold, c = learning rate, n = number of iterations
    def __init__(self, X, Y, wi, c, n):
        self.Weights = self.train(X, Y, wi, c, n)

    def train(self, X, Y, wi, c, n):
        w = np.zeros(3)
        w[0] = wi
        errors = []
        for t in range(n):  # entire data set. For large data sets, randomly pick a sample instead
            total_error = 0
            for i, x in enumerate(X):  # loop through each sample in data set, x = X[i]
                a = np.dot(x, w)
                if a >= 0:  # if a > threshold
                    d = 1
                else:
                    d = 0

                error = Y[i] - d
                w = w + c * x * error
                total_error += error

            errors.append(total_error * -1)

        plt.figure()
        plt.plot(errors)
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        return w

    def getWeights(self):
        return self.Weights


def testTraining(X, Y, w):
    mis = 0
    for i, x in enumerate(X):  # each sample in data set, x = X[i]
        a = np.dot(x, w)
        if a >= 0:  # if a > threshold
            d = 1
        else:
            d = 0

        if not d == Y[i]:
            mis += 1

    return mis

def testNetworks(data, Y1, w1, Y2, w2):
    error = 0
    for i, x in enumerate(data):
        a = np.dot(x, w1)
        if a >= 0: # versicolor or virginica (d = 1)
            error = testNet2(x, Y2, w2, i, error) # determine if this sample is versicolor or virginica and return error
        else: # setosa
            d = 0
            if not d == Y1[i]: # if d != Y, (y-d)^2 = 1
                error += 1
                print(i+1, ': satosa (incorrect)')
            else:
                print(i+1, ': satosa (correct)')

    return error


def testNet2(x, Y2, w2, i, error):
    # x = input data sample, Y2 = expected data, w2 = weights, i = iteration count, error = mean squared error

    a = np.dot(x, w2)
    if a >= 0: # virginica
        d = 1
        name = 'virginica'
    else: # versicolor
        d = 0
        name = 'versicolor'

    if not d == Y2[i-10]:
        error += 1
        print(i+1, ':', name, '(incorrect)')
    else:
        print(i+1, ':', name, '(correct)')
    return error


## data fetch and pre-processing

trnData = np.genfromtxt("train.txt", dtype=object, delimiter=",")

trnLen = len(trnData[:,0])  # number of training samples
train_data = np.zeros((trnLen, 3))  # training data using last 2 parameters
train_data[:, 0] = 1
train_data[:, 1] = trnData[:, 2]
train_data[:, 2] = trnData[:, 3]

train_labels = np.empty(trnLen, dtype=str)  # array of labels
train_labels = trnData[:, 4]
train_output1 = np.zeros(trnLen)  # expected output for setosa vs. other classification
train_output2 = np.zeros(trnLen-40)  # expected output for versicolor vs. virginica classification

# numClassData1: Setosa = 0, Other = 1
# numClassData2: Versicolor = 0, Virginica = 1
for i, name in enumerate(train_labels):
    name = name.decode('UTF-8')
    if name == 'Iris-setosa':
        train_output1[i] = 0
    elif name == 'Iris-versicolor':
        train_output1[i] = 1
        train_output2[i-40] = 0
    else:
        train_output1[i] = 1
        train_output2[i-40] = 1

# train network 1 (satosa vs. other classification)
network1 = PerceptronNetwork(train_data, train_output1, 0, 1, 20)
plt.title('Perceptron 1 Training')
plt.show()
w1 = network1.getWeights()

# train network 2 (versicolor vs. virginica classification)
network2 = PerceptronNetwork(train_data[40:], train_output2, -2000, 0.5, 20)
plt.title('Perceptron 2 Training')
plt.show()
w2 = network2.getWeights()

# plot data points and seperating lines
plt.figure()
plt.plot(train_data[:-80, 1], train_data[:-80, 2], '.b', label='Setosa')
plt.plot(train_data[40:-40, 1], train_data[40:-40, 2], '.r', label='Versicolor')
plt.plot(train_data[80:, 1], train_data[80:, 2], '.g', label='Virginica')
plt.plot([0, -w2[0]/w2[1]], [-w2[0]/w2[2], 0])
plt.plot([0, -w1[0]/w1[1]], [-w1[0]/w1[2], 0])
plt.axis('auto')
plt.legend()
plt.show()

# check to ensure training worked by feeding through testing data
print('Re-feeding training data for initial network test:')
mis1 = testTraining(train_data, train_output1, network1.getWeights())
mis2 = testTraining(train_data[40:], train_output2, network2.getWeights())
print('Network 1: {0}/40 incorrectly classified samples'. format(mis1))
print('Network 2: {0}/40 incorrectly classified samples'. format(mis2))

## Testing

# data fetch and pre-processing
tstData = np.genfromtxt("test.txt", dtype = object, delimiter = ",")
tstLen = len(tstData[:, 0]) # num of samples
test_data = np.zeros((tstLen, 3)) # testing data using last 2 parameters
test_data[:, 0] = 1
test_data[:, 1] = tstData[:, 2]
test_data[:, 2] = tstData[:, 3]

test_labels = np.empty(tstLen, dtype=str)
test_labels = tstData[:, 4] # array of labels
test_output1 = np.zeros(tstLen) # expected data for setosa vs. other classification
test_output2 = np.zeros(tstLen-10) # expected data for versicolor vs. virginica classification

# numClassData1: Setosa = 0, Other = 1
# numClassData2: Versicolor = 0, Virginica = 1
for i, name in enumerate(test_labels):
    name = name.decode('UTF-8')
    if name == 'Iris-setosa':
        test_output1[i] = 0
    elif name == 'Iris-versicolor':
        test_output1[i] = 1
        test_output2[i-10] = 0
    else:
        test_output1[i] = 1
        test_output2[i-10] = 1

# test networks and return total mean square
print('\nTesting classification:')
totalMeanSquare = testNetworks(test_data, test_output1, network1.getWeights(), test_output2, network2.getWeights())
print('Mean squared error =', totalMeanSquare)


