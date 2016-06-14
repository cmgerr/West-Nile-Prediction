import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('../Assets/train_cleaned.csv')
weather = pd.read_csv('../Assets/weather_clean.csv')

train.columns
weather.columns

merged = train.merge(weather, how = 'left', on = ['Station', 'Date'])

merged.shape

merged[['Station', 'Trap']].sort_values('Trap')

merged.Station.value_counts(0)
