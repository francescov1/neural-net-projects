import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import wavfile
import csv
data = pd.read_csv('sound.csv')
X = data.values
X[:, 0] -= np.mean(X[:, 0])
X[:, 1] -= np.mean(X[:, 1])

c = 0.2
weights = np.random.rand(X.shape[1], X.shape[1])
total_errors = list()
last_mse = 0
epoch = 0

#while True:
for n in range(1000):
    mse = 0
    for x in X:
        outputs = np.matmul(weights, x.transpose())

        K = outputs * outputs.reshape(2, 1)
        dw = c * (np.matmul(outputs, x) - np.matmul(K, weights))

        weights += dw
        mse += sum((x - np.matmul(weights.transpose(), outputs))**2)

    total_errors.append(mse)

    if np.mod(epoch, 100) == 0:
        print('Epoch', epoch, ': MSE =', mse)

    '''if epoch > 1 and (last_mse - mse) * 100 / mse < 0.01: # break out of loop if MSE doesnt change by more than 0.01%
        break'''

    epoch += 1
    last_mse = mse


print('Epochs:', epoch)
print('Final MSE:', mse)
print('Final weights:\n', weights)

Y = np.matmul(weights, X.transpose())
Y = Y.transpose()
y1 = Y[:, 0]
y2 = Y[:, 1]
wavfile.write('outputAudio1.wav', 8000, y1)
wavfile.write('outputAudio2.wav', 8000, y2)


file = open('output.csv', 'w')
writer = csv.writer(file, delimiter=',', quotechar='"')

writer.writerow(["Final weights"])
writer.writerow(["Output Node 1", "Output Node 2"])
for i in range(len(weights[0])):
    writer.writerow([weights[i, 0], weights[i, 1]])
writer.writerow(["Source 1", "Source 2"])

for i in range(len(y1)):
    writer.writerow([y1[i], y2[i]])

file.close()

plt.plot(total_errors)
plt.title('Error')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()



