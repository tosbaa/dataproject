import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import pandas as pd

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

data = pd.read_csv('HappinessAlcoholConsumption.csv')
data = np.array([data['HDI'], data['Spirit_PerCapita']])
data = np.stack((data[0], data[1]), axis=-1)


# plt.scatter(data.T[0], data.T[1], **plot_kwds)
# frame = plt.gca()
# frame.axes.get_xaxis().set_visible(False)
# frame.axes.get_yaxis().set_visible(False)
plt.xlabel("Happiness Score")
plt.ylabel("Spirit Drink Consumption")


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    print('Clustering took {:.2f} s'.format(end_time - start_time))


## K means
#plot_clusters(data, cluster.KMeans, (), {'n_clusters':4})
##

## Affinity Propagation
#plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-1000, 'damping':0.95})
##

##Mean Shift
#plot_clusters(data, cluster.MeanShift, (100,), {'cluster_all':False})
##

##Agloromerative
#plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})
##

##DBSCAN
#plot_clusters(data, cluster.DBSCAN, (), {'eps':30})
##

##HDBSCAN
#import hdbscan
#plot_clusters(data, hdbscan.HDBSCAN, (), {'min_cluster_size':4})
##

plt.show()



