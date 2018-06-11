import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
import csv

class Network:

    def __init__(self, n_inputs, n_hidden):
        self.hiddenLayer = np.random.rand(n_hidden+1, n_inputs+1) # +1 for bias
        self.outputLayer = np.random.rand(3, n_hidden+1)

    def train(self, inputs, expected_outputs, c, alpha, epsilon, epochs):
        mse = []

        for n in range(epochs):

            total_error = 0
            last_output_dws = np.zeros(self.outputLayer.shape)
            last_hidden_dws = np.zeros(self.hiddenLayer.shape)

            for s, sample in enumerate(inputs):
                hidden_outputs, outputs = self.forward_propogation(sample)
                hidden_dws, output_dws, error = self.backward_propagation(sample, hidden_outputs, outputs, expected_outputs[s], c, epsilon)
                self.outputLayer += (output_dws + alpha*last_output_dws)
                self.hiddenLayer += (hidden_dws + alpha*last_hidden_dws)
                total_error += error

                last_output_dws = output_dws
                last_hidden_dws = hidden_dws

            mse.append(total_error)

        plt.figure()
        plt.plot(mse)
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Error')
        plt.show()
        return total_error  # return last error

    # forward propagate one sample
    def forward_propogation(self, inputs):
        hidden_outputs = []
        hidden_outputs.append(1) # add 1 for bias weight input

        for i, neuron in enumerate(self.hiddenLayer[1:]):  # skip bias weight
            weights = neuron
            a = self.activation(inputs, weights)
            output = self.sigmoid(a)
            hidden_outputs.append(output)

        outputs = []
        for i, neuron in enumerate(self.outputLayer):
            weights = neuron
            a = self.activation(hidden_outputs, weights)
            output = self.sigmoid(a)
            outputs.append(output)

        a = self.activation(inputs, self.hiddenLayer[0])
        output = self.sigmoid(a)
        hidden_outputs[0] = output

        return hidden_outputs, outputs

    # takes in data for one sample and backpropagate errors
    def backward_propagation(self, inputs, hidden_outputs, outputs, expected_outputs, c, e):
        total_error = 0.0

        hidden_outputs = np.array(hidden_outputs)
        inputs = np.array(inputs)

        output_errors = []
        output_deltas = []
        output_dws = []

        # calculate output layer error
        for i, neuron in enumerate(self.outputLayer):
            d = expected_outputs[i]
            y = outputs[i]
            if d == 1-e and y >= d:
                error = 0
            elif d == e and y <= d:
                error = 0
            else:
                error = abs(d - y)

            output_errors.append(error)
            delta = error * self.sigmoidDer(y)
            output_deltas.append(delta)
            weight_change = c * hidden_outputs * delta
            weight_change[0] *= -1  # ensures dw for bias is substracted instead of added
            output_dws.append(weight_change)

            total_error += error**2

        hidden_errors = []
        hidden_deltas = []
        hidden_dws = []

        # calculate hidden layer error
        for i, neuron in enumerate(self.hiddenLayer):
            error = 0.0
            for j, output_neuron in enumerate(self.outputLayer):
                weight = output_neuron[i]  # get the weight from current hidden neuron to current output neurons
                error += weight * output_deltas[j]

            hidden_errors.append(error)
            delta = error * self.sigmoidDer(hidden_outputs[i])
            hidden_deltas.append(delta)
            weight_change = c * inputs * delta
            weight_change[0] *= -1  # ensures dw is substracted from bias weight instead of added
            hidden_dws.append(weight_change)

            total_error += error ** 2

        output_dws = np.array(output_dws)
        hidden_dws = np.array(hidden_dws)
        return hidden_dws, output_dws, total_error

    # test network
    def test(self, inputs, expected_outputs, e):

        outputs = []
        misclass = 0
        for s, sample_input in enumerate(inputs):
            _, sample_output = self.forward_propogation(sample_input)

            quality = self.round_output(sample_output, e)
            outputs.append(quality)

            for i, expected_val in enumerate(expected_outputs[s]):
                if quality[i] != expected_val:
                    misclass+=1
                    break

        accuracy = (len(expected_outputs) - misclass) * 100 / len(expected_outputs)

        return outputs, accuracy, self.hiddenLayer, self.outputLayer


    def activation(self, inputs, weights):
        return np.dot(inputs, weights)

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def sigmoidDer(self, output):
        return output * (1.0 - output)

    # rounds outputs to epsilon or 1-epsilon
    def round_output(self, output, e):
        output_rounded = [e] * len(output)
        index = output.index(max(output))
        output_rounded[index] = 1-e
        return output_rounded

# load data and specify pecentage used to train and validate (whatever is left will be testing)
def loadData(filename, per_train, per_validate):
    if per_train + per_validate >= 100:
        print('Error with percentage split')
        quit()

    data = pd.read_csv(filename)
    norm_data = normalize_dataframe(data)
    norm_data['bias'] = 1 # set bias inputs
    length = len(norm_data.index)
    n_train = length*per_train/100
    n_validate = length*per_validate/100

    train_data = norm_data.loc[:n_train, :]
    validate_data = norm_data.loc[n_train:(n_train+n_validate), :]
    test_data = norm_data.loc[(n_train+n_validate):, :]

    return train_data, validate_data, test_data

# convert outputs to binary (but use epsilon instead of 0 and 1)
def convert_expected_outputs(expected, e):
    expected_binary = []
    for quality in expected:
        if quality == 5:
            expected_binary.append([e, e, 1-e])
        if quality == 7:
            expected_binary.append([e, 1-e, e])
        if quality == 8:
            expected_binary.append([1-e, e, e])

    expected_binary = np.array(expected_binary)
    return expected_binary


# normalize input data
def normalize_dataframe(df):
    for col in df.iloc[:, :-1]: # don't normalize expected outputs
        df_col = df[[col]]
        x = df_col.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df[col] = pd.DataFrame(x_scaled)

    return df

# parameters for network
c = 0.7 # learning rate
alpha = 0.4 # momentum
epsilon = 0.2
epochs = 50

input_headers = ['bias', 'alcohol', 'density', 'total sulfur dioxide', 'chlorides']

train_data, validate_data, test_data = loadData('data.csv', 70, 15)

inputs_train = train_data.as_matrix(columns=input_headers)
inputs_validate = validate_data.as_matrix(columns=input_headers)
inputs_test = test_data.as_matrix(columns=input_headers)

expected_train_dt = train_data['quality']
expected_validate_dt = validate_data['quality']
expected_test_dt = test_data['quality']

expected_train = convert_expected_outputs(expected_train_dt, epsilon)
expected_validate = convert_expected_outputs(expected_validate_dt, epsilon)
expected_test = convert_expected_outputs(expected_test_dt, epsilon)

n_inputs = len(inputs_train[0]) - 1  # number of inputs in samples

network = Network(n_inputs, 4)

last_mse = network.train(inputs_train, expected_train, c, alpha, epsilon, epochs)
outputs_test, accuracy, hidden_weights, output_weights = network.test(inputs_test, expected_test, epsilon)

print('Final mean squared error =', last_mse)
print('Accuracy =', accuracy, '%')
print('Hidden Weights:', hidden_weights)
print('Final Weights:', output_weights)

file = open('output.csv', 'w')
writer = csv.writer(file, delimiter=',', quotechar='"')
expected_test = np.array(expected_test_dt) # get classes

writer.writerow(["Output", "Expected Output"])
for i, output_bin in enumerate(outputs_test):
    if output_bin == [epsilon, epsilon, 1-epsilon]:
        output = 5
    elif output_bin == [epsilon, 1-epsilon,epsilon]:
        output = 7
    elif output_bin == [1-epsilon, epsilon, epsilon]:
        output = 8
    else:
        output = 0

    writer.writerow([output, expected_test[i]])

file.close()
