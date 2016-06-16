import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

# define a random state to use throughout
rs = 8

# import data and define X, y (features, target)
df = pd.read_csv('../Assets/merged.csv')
df.info()
X = df.drop(['Date', 'Trap', 'Year','WnvPresent', 'date_station_id'], axis=1)
y = df['WnvPresent'].astype(int)

# select the 10 best features
kbest = SelectKBest(k=10)
X_kbest = kbest.fit_transform(X,y)

# get list of best feature names
best_features = [x for (x,y) in zip(X.columns, kbest.get_support().tolist()) if y==1]

best_features
# put feature names into dataframe (for reference)
X_kbest = pd.DataFrame(X_kbest, columns = best_features)


# do train_test_split on best features and target
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.2,
    stratify = y, random_state = rs)


# instantiate and fit LogisticRegression
lg = LogisticRegression(random_state=rs, n_jobs=-1, verbose=1)

# cross_val_score on all data
print cross_val_score(lg, X_kbest, y, n_jobs=-1, verbose=1).mean()

# fit model on training data and look at classification metrics
lg.fit(X_train, y_train)
lg.score(X_test, y_test)
lg_pred = lg.predict(X_test)

print confusion_matrix(y_test, lg_pred)

print classification_report(y_test, lg_pred)

# instantiate  RandomForestClassifier
rf = RandomForestClassifier(random_state=rs, n_jobs=-1, verbose=1, max_depth = None)

# look at cross_val_score of rf on all data
print cross_val_score(rf, X_kbest, y, n_jobs=-1, verbose=1).mean()

# fit model on training data and look at classification metrics
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

rf_pred = rf.predict(X_test)
print confusion_matrix(y_test, rf_pred)
print classification_report(y_test, rf_pred)
