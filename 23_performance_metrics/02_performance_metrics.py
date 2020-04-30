# import python package
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# input correct and predict data
X_actual = [5, -1, 2, 10]
Y_predic = [3.5, -0.9, 2, 9.9]
# proint perfroamce metrics
print ('R Squared =',r2_score(X_actual, Y_predic))
print ('MAE =',mean_absolute_error(X_actual, Y_predic))
print ('MSE =',mean_squared_error(X_actual, Y_predic))