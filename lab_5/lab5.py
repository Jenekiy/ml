import numpy as np
from Baes import Baes
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data 
Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)

normal = Baes()
normal.Parameters(X_train, y_train)
pred_train = normal.predict(X_train)
pred_test = normal.predict(X_test)
acc_train = normal.accuracy(y_train, pred_train)
acc_test = normal.accuracy(y_test, pred_test)
print(f"Нормальное распределение -> Train: {acc_train:.4f}, Test: {acc_test:.4f}")

normal.distribution = "laplace"
pred_train = normal.predict(X_train)
pred_test = normal.predict(X_test)
acc_train = normal.accuracy(y_train, pred_train)
acc_test = normal.accuracy(y_test, pred_test)
print(f"Лапласовское распределение -> Train: {acc_train:.4f}, Test: {acc_test:.4f}")

# Класс для визуализации
target_class = 0
class_idx = np.where(normal.classes == target_class)[0][0]
X_c = X_train[y_train == target_class]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
feature_names = iris.feature_names

for i in range(4):
    # Эмпирическое распределение (гистограмма реальных данных)
    axes[i].hist(X_c[:, i], bins=15, density=True, alpha=0.5, color='gray', label='Эмпирическое')
    
    # Генерация оси X
    x_min, x_max = axes[i].get_xlim()
    x_vals = np.linspace(x_min, x_max, 200)
    
    mean_val = normal.M[class_idx, i]
    var_val = normal.D[class_idx, i]
    
    # Нормальное распределение
    normal.distribution = 'normal'
    pdf_norm = normal.Model(x_vals, mean_val, var_val)
    axes[i].plot(x_vals, pdf_norm, 'r-', lw=2.5, label='Нормальное')
    
    # Лапласовское распределение
    normal.distribution = 'laplace'
    pdf_lap = normal.Model(x_vals, mean_val, var_val)
    axes[i].plot(x_vals, pdf_lap, 'b--', lw=2.5, label='Лапласовское')

    axes[i].set_title(f'Класс {target_class}\n{feature_names[i]}')
    axes[i].set_xlabel('Значение (см)')
    axes[i].set_ylabel('Плотность')
    axes[i].legend()

plt.tight_layout()
plt.show()