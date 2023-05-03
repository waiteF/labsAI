import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Зчитуємо дані з файлу labs01.csv і перетворюємо їх на numpy масив
data = pd.read_csv('labs01.csv', header=None).values

# Візуалізуємо вихідні дані (точки на площині)
plt.scatter(data[:, 0], data[:, 1])
plt.title('Вихідні точки на площині')
plt.show()

# Метод зсуву середнього (знаходження оптимальної кількості кластерів)
distortions = []
for i in range(1, 16):
    kmeans_model = KMeans(n_clusters=i)
    kmeans_model.fit(data)
    distortions.append(kmeans_model.inertia_)

# Знаходимо точку згину графіка (методом "ліктя")
elbow_point = np.argmin(np.diff(distortions)) + 1

# Візуалізуємо результати методу зсуву середнього (центри кластерів)
kmeans_model = KMeans(n_clusters=elbow_point)
kmeans_model.fit(data)
centers = kmeans_model.cluster_centers_
plt.scatter(data[:, 0], data[:, 1])
plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='r', s=150)
plt.title('Центри кластерів (метод зсуву середнього)')
plt.show()

# Обчислюємо score для різних кількостей кластерів (від 2 до 15)
scores = []
for i in range(2, 16):
    kmeans_model = KMeans(n_clusters=i)
    kmeans_model.fit(data)
    score = silhouette_score(data, kmeans_model.labels_)
    scores.append(score)

# Візуалізуємо результати оцінки score
plt.bar(range(2, 16), scores)
plt.title('Бар діаграмма score(number of clusters)')
plt.show()

# Кластеризуємо дані методом k-середних з оптимальною кількістю кластерів
kmeans = KMeans(n_clusters=num_clusters_optimal, init=centroids_optimal, max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Виводимо графіки
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('KMeans Clustering', fontsize=20)

# Рисуємо вихідні точки на площині
axs[0, 0].scatter(X[:,0], X[:,1], c='black')
axs[0, 0].set_title('Input data')

# Рисуємо центри кластерів (метод зсуву середнього)
axs[0, 1].scatter(X[:,0], X[:,1], c=pred_y)
axs[0, 1].scatter(centers[:,0], centers[:,1], s=100, c='red')
axs[0, 1].set_title('Shifted Mean Centroids')

# Рисуємо бар діаграмму score(number of clusters)
axs[1, 0].bar(range(2, max_clusters + 1), scores)
axs[1, 0].set_title('Cluster Scores')
axs[1, 0].set_xlabel('Number of clusters')
axs[1, 0].set_ylabel('Score')

# Рисуємо кластеризовані дані з областями кластеризації
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']
for i in range(num_clusters_optimal):
axs[1, 1].scatter(X[pred_y == i, 0], X[pred_y == i, 1], s=30, c=colors[i], label=f'Cluster {i+1}')
axs[1, 1].scatter(centers[:,0], centers[:,1], s=100, c='black', marker='X')
axs[1, 1].set_title(f'KMeans Clustering (Optimal {num_clusters_optimal} clusters)')
axs[1, 1].legend()

# Показуємо графіки
plt.show()