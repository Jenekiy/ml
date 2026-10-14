import numpy as np
from XOR import XOR

@staticmethod
def Relu(z):
    return np.maximum(0.0, z)

@staticmethod
def Relu_grad(z):
    return (z > 0).astype(float)

@staticmethod
def Sigma(z):
    return 1 / (1 + np.exp(-z))

@staticmethod
def Sigma_grad(x):
    sig = Sigma(x)
    return sig * (1.0 - sig)

@staticmethod
def Linear(x):
    return x

@staticmethod
def Linear_grad(x):
    return 1

def main():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    for i in range(3):
        net = 0
        if (i == 0):
            net = XOR(X, y, Relu, Relu_grad, net=[5, 1], seed = 1)
        elif (i == 1):
            net = XOR(X, y, Sigma, Sigma_grad, net=[5, 1], seed = 1)
        else:
            net = XOR(X, y, Linear, Linear_grad, net=[5, 1], seed = 1)

        print("=== До обучения ===")
        for xi, pi, yi in zip(X, net.predict(), y):
            print(f"x={xi}  pred={pi[0]: .4f}  y={yi[0]}")

        print("\n=== Обучение ===")
        net.BackProp(lr=0.05)

        print("\n=== После обучения ===")
        for xi, ai, yi in zip(X, net.predict(), y):
            ok = "OK" if abs(ai[0] - yi[0]) < 0.5 else "FAIL"
            print(f"x={xi}  pred={ai[0]: .4f}  y={yi[0]}  {ok}")

        print(f"\nQ (последняя эпоха) = {net.Q:.6f}")


main()