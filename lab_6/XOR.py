import numpy as np


class XOR:
    def __init__(self, x, y, act, act_grad, net=[5, 1], seed=None):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

        self.net = [self.x.shape[1]] + list(net)
        self.l = len(self.net) - 1

        rng = np.random.default_rng(seed)

        self.w = []   # список матриц
        self.b = []   # список векторов

        self.act = act
        self.act_grad = act_grad

        for i in range(self.l):
            H_in  = self.net[i]
            H_out = self.net[i + 1]

            limit = 1.0 / np.sqrt(H_in)
            W = rng.uniform(-limit, limit, size=(H_in, H_out))
            b = np.zeros(H_out)

            self.w.append(W)
            self.b.append(b)

        self.Q = 0.0

    def predict(self, X=None):
        if X is None:
            X = self.x
        a = np.asarray(X, dtype=float)

        for i in range(self.l):
            z = a @ self.w[i] + self.b[i]
            a = self.act(z) if i < self.l - 1 else z
        return a
    
    def Q_init(self):
        self.Q = 0.0

    def accumulate_Q(self, pred, true):
        self.Q += float(np.sum((pred - true) ** 2))

    def BackProp(self, lr=0.01, epochs=3000):
        for epoch in range(epochs):
            self.Q_init()
            for xi, yi in zip(self.x, self.y):
                self._backprop_one(xi, yi, lr)

    def _backprop_one(self, x, y, lr):
        a = x.reshape(-1)
        activations = [a]
        zs = []

        for i in range(self.l):
            z = self.w[i].T @ a + self.b[i]
            zs.append(z)

            if i < self.l - 1:
                a = self.act(z)
            else:
                a = z

            activations.append(a)

        self.accumulate_Q(a, y)
        eps = a - y 
        for i in reversed(range(self.l)):
            a_prev = activations[i] 

            gw = np.outer(a_prev, eps)
            gb = eps.copy()

            if i > 0:
                d_prev = self.w[i] @ eps
                eps = d_prev * self.act_grad(zs[i - 1])

            self.w[i] -= lr * gw
            self.b[i] -= lr * gb    