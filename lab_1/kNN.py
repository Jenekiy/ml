from math import sqrt
import matplotlib.pyplot as plt
class kNN:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.arr = {}
        for x_l, y_i in zip(self.x, self.y):
            self.arr[tuple(x_l)] = y_i 
            
    # Функция ядра Епанечникова
    def Kernel(self, r):
        return 0.75 * (1 - r**2) * (1 if r <= 1 else 0) 
        
    # Метрика (Евклидова)
    def Dist(self, a, b):
        sum = 0
        for i in range(len(a)): 
            sum += (a[i] - b[i])**2
        return sqrt(sum)
        
    # Массив ближайших объектов
    def K_Sort(self, x_i, k):
        fn = list(self.x)
        fn.sort(key=lambda x_j: self.Dist(x_i, x_j)) 
        return fn[:k + 2]
        
    # Количество ошибок
    def LOO(self):
        mn = 1e10
        mb_k = -1
        history = []
        for k_i in range(1, 30):
            sum_err = 0
            for x_i in self.x:
                if self.a(x_i, k_i) != self.arr[tuple(x_i)]: 
                    sum_err += 1
            history.append((k_i, sum_err))
            if (mn > sum_err):
                mn = sum_err
                mb_k = k_i
        return mb_k, mn, history
        
    # Метрический классификатор
    def a(self, m, k):
        mx = -1
        mb_y = -1
        dist_x = self.K_Sort(m, k)
        
        h = self.Dist(m, dist_x[-1]) 
        if h == 0: h = 1e-5

        for y_i in set(self.y): 
            sum = 0
            for i in range(1, len(dist_x) - 1): 
                if (y_i == self.arr[tuple(dist_x[i])]):
                    sum += self.Kernel(self.Dist(m, dist_x[i]) / h) 
            if (sum > mx):
                mx = sum
                mb_y = y_i
        return mb_y  
        
    # Вывод  
    def Param(self):
        best_k, min_loo, history = self.LOO()
        accuracy = (len(self.x) - min_loo) / len(self.x)

        ks = [h[0] for h in history]
        errs = [h[1] for h in history]
        
        plt.figure(figsize=(10, 5))
        plt.plot(ks, errs, marker='o', color='b')
        plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
        plt.title('График зависимости ошибок LOO от k')
        plt.xlabel('Параметр k')
        plt.ylabel('Количество ошибок')
        plt.legend()
        plt.grid(True)
        plt.show()
        # -------------------------
        
        return { 
            "Оптимальный K" : best_k, 
            "Кол-во ошибок LOO" : min_loo, 
            "Accuracy" : round(accuracy, 4) 
        }
    