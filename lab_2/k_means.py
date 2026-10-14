import numpy as np
from math import sqrt
class k_Means:
    def __init__(self, usl, k):
        self.usl = np.array(usl)
        self.k = k
        self.means = self.Center_Init(self.usl, k)
        self.a = None # Словарь кластеров
        self.labels = None # Массив меток для удобства отрисовки

    # Евклидово расстояние
    def Dist(self, x_i, x_j):
        return np.linalg.norm(x_i - x_j)

    # Инициализация центров
    def Center_Init(self, usl, k):
        idx = np.random.choice(len(usl), k, replace=False)
        return usl[idx]

    # Кластеризация
    def Klast(self): 
        a = {i: [] for i in range(self.k)} # Создаем пустые списки для всех k
        labels = []
        for x_i in self.usl:
            mn = float('inf')
            a_i = -1
            for j in range(self.k):
                d = self.Dist(x_i, self.means[j])
                if d < mn:
                    mn = d
                    a_i = j
            a[a_i].append(x_i)
            labels.append(a_i)
        return a, labels

    # Новые средние кластеров
    def Center_New(self):
        self.a, self.labels = self.Klast()
        new_means = np.zeros_like(self.means)
        
        for a_i, arr in self.a.items():
            if len(arr) > 0:
                new_means[a_i] = np.mean(arr, axis=0) 
        return new_means

    # Выполнение алгоритма Ллойда
    def Exec(self):
        max_iters = 100
        for _ in range(max_iters):
            new_means = self.Center_New()
            if np.allclose(self.means, new_means):
                break
            self.means = new_means

    # Сумма средних внутрикластерных расстояний
    def F0(self):
        res = 0
        for i in range(self.k):
            if len(self.a[i]) == 0: continue
            u_a = self.means[i]
            sum_dist = 0
            for x_i in self.a[i]:
                sum_dist += self.Dist(x_i, u_a)
            res += sum_dist / len(self.a[i])
        return res

    # Сумма межкластерного расстояния
    def F1(self):
        res = 0
        for u_a in self.means:
            for u_b in self.means:
                res += self.Dist(u_a, u_b)
        return res

    # Качество кластеризации
    def Quality(self):
        f0 = self.F0()
        if f0 == 0: 
            return 0
        return self.F1() / f0