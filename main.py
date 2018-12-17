import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#ffmc influences ignition and fire spread
#DMC and dc influence fire intensity
#ISI correlates with fire velocity spread
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')
#df = pd.read_csv("C:\\Users\\balli\\Downloads\\forestfires.csv")
df['area'] = df['area'].apply(np.log1p)
hmcol = df[['X', 'Y', 'area']]
xyareamean = hmcol.groupby(['X', 'Y']).mean()
xyareamean = xyareamean.reset_index()

#histogram frequency of area
plt.hist(df['area'])
plt.xlabel('are', size=15)
plt.ylabel('Frequency', size=15)
#plt.show()
plt.close()

#heatmap average fire size
plt.figure(figsize=(8,8))
pivot_table = xyareamean.pivot('Y', 'X',)['area']
plt.xlabel('X', size=15)
plt.ylabel('Y', size=15)
plt.title('Average Size of Burnt Area', size=15)
sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Reds')
#plt.show()
plt.close()

#heatmape fire frequency
xycomb = df.groupby(['X', 'Y']).size().reset_index(name="Freq")
plt.figure(figsize=(8,8))
pivot_table = xycomb.pivot('Y', 'X',)['Freq']
pivot_table = pivot_table.round()
plt.xlabel('X', size=15)
plt.ylabel('Y', size=15)
plt.title('Frequency of Fires', size=15)
sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Reds')
#plt.show()
plt.close()

#frequency of fire in months/days
weekdays = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
mapw = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
mapm = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
count = df['month'].value_counts()
count.index = pd.CategoricalIndex(count.index, categories=months, ordered=True)
tryhard = count.sort_index().plot('bar')
plt.title('Frequency of Fires', size=15)
#plt.show()
plt.close()


#regression attempt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
X = df[['temp', 'RH', 'wind', 'rain']]
#X = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]
X = preprocessing.scale(X)
y = df['area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=2**-5, gamma=0.1)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
y_rbf = np.expm1(y_rbf)
plt.scatter(y_test, y_rbf)
print(mean_squared_error(y_test, y_rbf))
print(r2_score(y_test, y_rbf))
plt.xlabel('Actual', size=15)
plt.ylabel('Predicted', size=15)
#plt.show()
plt.close()

#clustering elbow method
from sklearn.cluster import KMeans
distortions = []
X = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']]
X = df[['FFMC', 'DMC', 'DC', 'ISI']]
X = df[['RH', 'wind', 'rain', 'temp']]
for i in range (1, 16):
   km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=0)
   y_km = km.fit_predict(X)
   distortions.append(km.inertia_)
plt.plot(range(1, 16), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method RH, wind, rain, temp')
plt.show()
plt.close()

#clustering silhouette plot
km = KMeans(n_clusters=4, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=0)
y_km = km.fit_predict(X)
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
   c_silhouette_vals = silhouette_vals[y_km == c]
   c_silhouette_vals.sort()
   y_ax_upper += len(c_silhouette_vals)
   color = cm.jet(i / n_clusters)
   plt.barh(range(y_ax_lower, y_ax_upper),
   c_silhouette_vals,
   height=1.0,
   edgecolor='none',
   color=color)
   yticks.append((y_ax_lower + y_ax_upper) / 2)
   y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.title('RH, wind, rain, temp')
plt.show()
plt.close()

#this transforms area into 5 classes
df['area'] = df['area'].round()

#comment these lines in for 2 classes
#di = {2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1, 6.0: 1, 7.0: 1}
#df['area'] = df['area'].replace(di)

#plot to see classes
df = df[(df[['area']] < 5).all(axis=1)]
df['area'].value_counts().plot('bar')
plt.xlabel('area', size=15)
plt.ylabel('Frequency', size=15)
plt.show()
plt.close()


#scatterplot matrix with classes
dfpart = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]
pd.plotting.scatter_matrix(dfpart, figsize=(6,6), c=df['area'])
plt.show()
plt.close()

#chi squared values
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
df['month'] = df['month'].map(mapm)
df['day'] = df['day'].map(mapw)
X = df[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]
y = df['area']
sel = SelectKBest(chi2, k=2)
X_new = sel.fit_transform(X, y)
kbestscores = pd.DataFrame(data=sel.scores_, index=X.dtypes.index)
objects = ('X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain')
y_pos = np.arange(len(objects))
plt.bar(y_pos, sel.scores_)
plt.xticks(y_pos, objects)
plt.xlabel('Feature', size=15)
plt.ylabel('Score', size=15)
plt.title('Chi^2 Test')
plt.show()
plt.close()


#scatterplot with classes
dfpart = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]
dfpart = pd.DataFrame(dfpart)
pd.plotting.scatter_matrix(dfpart, figsize=(6,6), c=df['area'])
plt.show()
plt.close()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X = df[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']]
y = df['area']


for i in range(2, 8):
    sel = SelectKBest(chi2, k=i)
    X_new = sel.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=0)
    pipe_tree = Pipeline([('scl', StandardScaler()), ('tree', DecisionTreeClassifier(random_state=0))])
    pipe_knn = Pipeline([('scl', StandardScaler()), ('knn', KNeighborsClassifier())])
    pipe_svc = Pipeline([('scl', StandardScaler()), ('svc', SVC(random_state=0))])
    param_grid_tree = [{'tree__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
    param_grid_knn = [{'knn__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}]
    param_grid_svc = [{'svc__C': [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15],
                       'svc__gamma': [2**3, 2**1, 2**-1, 2**-3, 2**-5, 2**-7, 2**-9, 2**-11, 2**-13, 2**-15],
                       'svc__kernel': ['rbf']},
                      {'svc__C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000],
                       'svc__kernel': ['linear']}]
    clf = GridSearchCV(estimator=pipe_svc, param_grid=param_grid_svc, cv=10, scoring='accuracy', n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    print('k=%d %.3f > %.3f %r' % (i, clf.best_score_, accuracy_score(y_test, clf.predict(X_test)), clf.best_params_))
