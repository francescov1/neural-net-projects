import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data.csv')
corr = data.corr()
print(corr)

inputs = data.as_matrix(columns=['alcohol', 'density', 'total sulfur dioxide', 'chlorides'])
expected_outputs = data["quality"]
expected_outputs = expected_outputs.as_matrix()
