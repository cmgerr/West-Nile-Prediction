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
# delete address fields because location has already been translated
# into latitude and longitude
del train['Address']
del train['Block']
del train['Street']
del train['AddressNumberAndStreet']
del train['AddressAccuracy']

# aggregate observations that are only distinct because of
# hitting the 50 mosquito cap
grouped = train.groupby(['Date', 'Species', 'Trap', 'Latitude', 'Longitude'])
aggregated = pd.DataFrame(grouped.agg({'NumMosquitos': np.sum, 'WnvPresent': np.max})).reset_index()
aggregated.sort_values(by = 'NumMosquitos', ascending = False)
# Sort by date and change to datetime
aggregated.sort_values(by='Date', inplace=True)
# set index to date
aggregated.set_index('Date')
# convert to datetime
aggregated.index = aggregated.index.to_datetime()
# add month column
aggregated['Month'] = aggregated.index.month
aggregated['Year'] = aggregated.index.year

# check if mosquito vars should be categorical
aggregated.Species.value_counts()

# create dummy vars for mosquito types
aggregated = pd.get_dummies(aggregated, columns = ['Species'])



# look at dispersion of traps geographically

plt.scatter(aggregated.Longitude, aggregated.Latitude)

# look at dispersion of virus incidence geographically
plt.scatter(aggregated.Longitude, aggregated.Latitude, c = aggregated.WnvPresent, alpha = .05)

# look at distribution of number of mosquitos
plt.hist(aggregated.NumMosquitos, 50)


# look at incidence of cases over time
aggregated[['Date', 'WnvPresent']].groupby('Date').sum().plot(kind='bar')

# need to further explore seasonality of the virus
