# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
# from sklearn.linear_model import LogisticRegression

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        # weights initialization
        self.theta = np.zeros(X.shape[1])

    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    def predict_prob(self, X):
        if self.fit_intercept:
            print ('p1 => X[:10]:\n', X[:10])
            # add extra dimension 
            # https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
            X = self.__add_intercept(X)
            #print ('p2 => X[:10]:\n', X[:10])
        print ('p3 => self.theta[:10]:\n', self.theta[:10])
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

# load the dataset from sklearn into local drive:
# C:\Users\14088\AppData\Local\Programs\Python\Python38-32
# \Lib\site-packages\sklearn\datasets\data\iris.csv
# I copy to the same location of 01_binary_logic.py
iris = datasets.load_iris()
print ('iris:\n', iris)
data = iris.data
print ('data.shape:', data.shape)
print ('data[:, 0].shape:\n', data[:, 0].shape)
print ('data[:, 0]:\n', data[:, 0])
print ('data[:, 1].shape:\n', data[:, 1].shape)
print ('data[:, 1]:\n', data[:, 1])

X = iris.data[:, :2]
print ('X[:3, :2]:\n', X[:3, :2])

# [0, 0, 0, 2, 2] => iris.target !=0 => false or true => * 1 
# => [0, 0, 0, 1, 1]
# he following code change the non-zero (e.g., 1,2, 3, etc) 
# into 0 or 1. (first, convert into true and then 1)
y = (iris.target != 0) * 1

print ('iris.target[:3]:\n', iris.target[:3])
print ('y:\n', y)
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='g', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='y', label='1')
# plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='r', label='2')
plt.legend();
plt.show()

# https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
model = LogisticRegression(lr = 0.1, num_iter = 300000)
# model.fit(X, y)
#print ('X[:3]', X[:3])
preds = model.predict(X)
(preds == y).mean()

plt.figure(figsize = (10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color = 'g', label = '0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color = 'y', label = '1')
plt.legend()

x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_prob(grid).reshape(xx1.shape)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='red');
plt.show()
