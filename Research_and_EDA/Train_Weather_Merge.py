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

# Cleaning
# Date column to date_time
merged['Date'] = pd.to_datetime(merged.Date)

# Date column to index
merged.set_index('Date', inplace=True)

# Since test set includes 2008, 2010, 2012, and 2014, pull out these years from weather
merged = merged[(merged.index.year == 2007) | (merged.index.year == 2009) | 
                (merged.index.year == 2011) | (merged.index.year == 2013)]

# EDA
# Identify number of WNV cases observed each year
merged[['WnvPresent']].resample('A').apply(sum)

# Plot number of WNV cases each month, with each year
# First create dataframes for each year
merged_2007 = merged['2007']
merged_2009 = merged['2009']
merged_2011 = merged['2011']
merged_2013 = merged['2013']

# Plot using subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 7))
ax0, ax1, ax2, ax3 = axes.flat

ax0.plot(merged_2007[['WnvPresent']].resample('M').apply(sum))
ax1.plot(merged_2009[['WnvPresent']].resample('M').apply(sum))
ax2.plot(merged_2011[['WnvPresent']].resample('M').apply(sum))
ax3.plot(merged_2013[['WnvPresent']].resample('M').apply(sum))

plt.tight_layout()

# 