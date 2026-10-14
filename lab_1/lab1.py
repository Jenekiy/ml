from sklearn.datasets import load_iris
from kNN import kNN
iris = load_iris()
X = iris.data 
Y = iris.target 

ad = kNN(X, Y)
print(ad.Param())