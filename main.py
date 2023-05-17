from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# http://www.sthda.com/english/wiki/wiki.php?id_contents=7932

# Se genera un conjunto de datos de prueba con 4 clusters. Se le indica al computador que detecte el numero de grupos usando el metodo del coeficiente de silueta, con un valor maximo de 10 clusters.

# Generamos los datos
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Graficamos los datos
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

def find_optimal_clusters(X, max_k):
    # utilizamos el metodo del coeficiente de silueta
    silhouette_scores = []
    for k in range(2, max_k):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    # Graficamos el coeficiente de silueta en función del número de clusters K
    if False:
        plt.plot(range(2, 10), silhouette_scores)
        plt.xlabel('Número de clusters')
        plt.ylabel('Coeficiente de silueta')
        plt.show()

    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_k


# Encontramos el número óptimo de clusters
optimal_k = find_optimal_clusters(X, 10) # se prueban 10 grupos como maximos

# Definimos el número de clusters
# k = 4 # este valor se obtiene del metodo de coeficiente de silueta

k = optimal_k

# Creamos el modelo de KMeans
kmeans = KMeans(n_clusters=k)

# Entrenamos el modelo con los datos
kmeans.fit(X)

# Obtenemos las etiquetas de los clusters y los centros
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Graficamos los datos con diferentes colores para cada cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# Graficamos los centros de los clusters
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
