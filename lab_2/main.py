from k_means import k_Means
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data 
k = len(np.unique(iris.target)) 

mx_e = -1
best_model = None

for i in range(10):
    ad = k_Means(X, k)
    ad.Exec()
    e = ad.Quality()
    
    if e > mx_e:
        mx_e = e
        best_model = ad

print(f"Лучшее качество (F1/F0): {mx_e:.2f}")
# 0: sepal length (длина чашелистика)
# 1: sepal width (ширина чашелистика)
f1, f2 = 0, 1 

plt.figure(figsize=(10, 6))

plt.scatter(X[:, f1], X[:, f2], c=best_model.labels)

# Рисуем найденные алгоритмом центры кластеров
centers = best_model.means
plt.scatter(centers[:, f1], centers[:, f2], 
            c='red', marker='x', s=250, linewidths=3, 
            label='Найденные центры (K-Means)')

plt.title('Кластеризация K-Means на датасете Iris')
plt.xlabel(iris.feature_names[f1])
plt.ylabel(iris.feature_names[f2])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

