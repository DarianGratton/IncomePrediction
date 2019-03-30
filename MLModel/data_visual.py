import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import load_data as loader

# Load the data
data = pd.read_csv("./trainingset.csv")
print('\n\ndata.info():')
print(data.info())
