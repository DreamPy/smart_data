from sklearn import  cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import seaborn as sns
import pandas as pd

def  _use_kmeans(data,is_vis,n_clusters=3):
    _cluster = cluster.KMeans(n_clusters=n_clusters)
    model = _cluster.fit(data)
    return {"labels": model.labels_.tolist(),"cluster_centers":model.cluster_centers_.tolist()}
def _use_tsne(X,n_components=2):
    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(data)
    return X_tsne

def _scatter_2(X,labels,x_name='x',y_name='y'):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    _data = pd.DataFrame(X,columns=[x_name,y_name],index= range(X.shape[0]))
    _data['labels'] = labels
    fg = sns.FacetGrid(data=_data,hue='labels',size=8).map(plt.scatter,x_name,y_name).add_legend()
    return fg.ax
