import numpy as np
class Baes:
    def __init__(self, distribution='normal'):
        self.M = None # Матрица средних
        self.D = None # Матрица дисперсий
        self.Med = None # Матрица медиан
        self.priors = None # Априорные вероятности классов
        self.distribution = distribution # Выбор модели
    def Parameters(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.M = np.zeros((n_classes, n_features))
        self.D = np.zeros((n_classes, n_features))
        self.Med = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.M[idx, :] = np.mean(X_c, axis=0)
            self.D[idx, :] = np.var(X_c, axis=0)
            self.Med[idx, :] = np.median(X_c, axis=0)
            self.priors[idx] = X_c.shape[0] / X.shape[0]
    def Model(self, x, mean, var):
        var = var + 1e-9 
        
        if self.distribution == 'normal':
            return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
        
        elif self.distribution == 'laplace':
            b = np.sqrt(var / 2.0)
            return (1.0 / (2 * b)) * np.exp(-np.abs(x - mean) / b)
        
    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                if self.distribution == 'laplace':
                    center = self.Med[idx, :]
                else:
                    center = self.M[idx, :]
                vals = self.Model(x, center, self.D[idx, :])

                sum = np.sum(np.log(vals + 1e-9))
                
                posteriors.append(prior + sum)

            y_pred.append(self.classes[np.argmax(posteriors)])
            
        return np.array(y_pred)
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)