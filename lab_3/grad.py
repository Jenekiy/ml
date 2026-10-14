import numpy as np

class RidgeSGD:
    def __init__(self):
        self.w = None
        self.Q_history = []
    # Скалярное произведение (а точнее матричное) - само предсказание
    def Skalyar(self, X):
        return np.dot(X, self.w)
    # Коэффициент детерминации
    def R2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        D = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / D)
    # Инициализация оценки Q
    def Q_init(self, t, X, Y):
        l = X.shape[0]
        len = max(5, int(l * 0.5)) 
        index = np.random.choice(l, len, replace=False)
        X = X[index]
        Y = Y[index]
        margin = np.dot(X, self.w) - Y
        mean_loss = np.mean(margin**2)
        return mean_loss + (t / 2) * np.sum(self.w**2)
    # Стохастичесий градиентный спуск 
    def Grad_Spusk(self, X, Y, t, lam=0.1, eps=1e-2, max_iter=100000):
        l, n = X.shape
        self.w = np.zeros(n)
        self.Q_history = []
        Q = self.Q_init(t, X, Y)
        for i in range(1, max_iter + 1):
            Q_prev = Q
            h = 1.0 / i     
            idx = np.random.randint(0, l)
            x_i = X[idx]
            y_i = Y[idx] 
            margin = np.dot(self.w, x_i) - y_i 

            self.w = self.w * (1 - h * t) - h * (margin * x_i)

            error = margin**2 + (t / 2) * np.sum(self.w**2)

            Q = lam * error + (1 - lam) * Q_prev

            self.Q_history.append(Q)
            if abs(Q - Q_prev) < eps:
                print(f"Сошлось на итерации {i}")
                break