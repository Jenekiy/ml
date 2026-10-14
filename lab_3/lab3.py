import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from grad import RidgeSGD
data = load_diabetes()
X = data.data
y = data.target

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X = np.hstack([X, -np.ones((X.shape[0], 1))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = RidgeSGD()
base_model.Grad_Spusk(X_train, y_train, t=0.01, eps=1e-3)

plt.figure(figsize=(10, 5))
plt.plot(base_model.Q_history[:2000])
plt.title("График сходимости функции потерь $Q_i$")
plt.xlabel("Номер итерации (i)")
plt.ylabel("Значение Q (Скользящее среднее)")
plt.grid(True)
plt.show()

print("Оптимизация параметра регуляризации tau...")
# Массив параметров t
t_arr = np.logspace(-4, 2, 10) 
R2 = []
best_t = None
best_r2 = -float('inf')

for t in t_arr:
    model = RidgeSGD()
    # Обучаем
    model.Grad_Spusk(X_train, y_train, t=t, eps=1e-3)
    
    # Тестим
    y_pred = model.Skalyar(X_test)
    
    # Считаем R2
    r2 = model.R2(y_test, y_pred)
    R2.append(r2)
    if r2 > best_r2:
        best_r2 = r2
        best_t = t

print(f"Лучший tau: {best_t:.5f}, Лучший R^2: {best_r2:.4f}")

# Построение графика зависимости R^2 от t
plt.figure(figsize=(10, 5))
plt.plot(t_arr, R2, marker='o', markersize=4)
plt.title("Зависимость качества $R^2$ на тестовой выборке от параметра $\\tau$")
plt.xlabel("Параметр регуляризации $\\tau$ (log scale)")
plt.ylabel("Коэффициент детерминации $R^2$")
plt.axvline(best_t, color='red', linestyle='--', label=f'Лучший $\\tau$ = {best_t:.4f}')
plt.legend()
plt.grid(True)
plt.show()