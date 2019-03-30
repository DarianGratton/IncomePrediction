import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import load_data as loader

# Load the data
data = loader.load_full_data()
print('\n\ndata.info():')
print(data.info())

print(type(data))
