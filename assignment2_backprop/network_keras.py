from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing


def loadData(filename, per_train):
    data = pd.read_csv(filename)
    norm_data = normalize_dataframe(data)

    length = len(norm_data.index)
    n_train = length*per_train/100

    train_data = norm_data.loc[:n_train, :]
    test_data = norm_data.loc[n_train:, :]

    return train_data, test_data


def normalize_dataframe(df):
    for col in df.iloc[:, :-1]:  # skip quality column
        df_col = df[[col]]
        x = df_col.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df[col] = pd.DataFrame(x_scaled)
    return df

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



# fix random seed for reproducibility
np.random.seed(7)

data, data_test = loadData('data.csv', 70)

input_headers = ['alcohol', 'density', 'total sulfur dioxide', 'chlorides']
epsilon = 0

inputs = data.as_matrix(columns=input_headers)
inputs_test = data_test.as_matrix(columns=input_headers)
expected_dt = data['quality']
expected_test_dt = data_test['quality']
expected = convert_expected_outputs(expected_dt, epsilon)
expected_test = convert_expected_outputs(expected_test_dt, epsilon)


model = Sequential()

# add layers

model.add(Dense(4, activation='sigmoid', input_dim=4))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])  # opt = 'rmsprop'

# Convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(expected, num_classes=3)

# iterate training data in batches
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(inputs, expected, epochs=100, batch_size=32)


# performance eval
loss_and_metrics = model.evaluate(inputs_test, expected_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], loss_and_metrics[1]*100))

# generate predictions on new data
#classes = model.predict(x_test, batch_size=128)