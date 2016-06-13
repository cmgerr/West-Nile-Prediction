import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.style.use('ggplot')

# read data into DataFrame
train = pd.read_csv('../Assets/train.csv')

# look at data
train.head()

# check dtypes and null values
train.info()

# Sort by date and change to datetime
train.sort_values(by='Date', inplace=True)
# set index to date
train.set_index('Date')
# convert to datetime
train.index = train.index.to_datetime()
# add month column
train['Month'] = train.index.month
train['Year'] = train.index.year
# check if mosquito vars should be categorical
train.Species.value_counts()

# create dummy vars for mosquito types
train = pd.get_dummies(train, columns = ['Species'])

# delete address fields because location has already been translated
# into latitude and longitude
del train['Address']
del train['Block']
del train['Street']
del train['AddressNumberAndStreet']
del train['AddressAccuracy']

# look at dispersion of traps geographically

plt.scatter(train.Longitude, train.Latitude)

# look at dispersion of virus incidence geographically
plt.scatter(train.Longitude, train.Latitude, c = train.WnvPresent, alpha = .05)

# look at distribution of number of mosquitos
plt.hist(train.NumMosquitos, 50)


# look at incidence of cases over time
train[['Date', 'WnvPresent']].groupby('Date').sum().plot(kind='bar')

# need to further explore seasonality of the virus
# we also need to aggregate observations that are only separate because of
# having passed the 50 mosquito threshold - next steps!
