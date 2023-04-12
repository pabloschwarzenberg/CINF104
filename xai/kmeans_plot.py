#Extraído de los ejemplos del libro Géron, A. (2022). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. " O'Reilly Media, Inc.".
#https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb
import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, axes):
    axes.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, axes, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    axes.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    axes.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, axes, title,resolution=1000):
    mins = [-50,-50]
    maxs = [50,50]
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    axes.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    axes.set_title(title)
    plot_data(X,axes)
    plot_centroids(clusterer.cluster_centers_,axes)