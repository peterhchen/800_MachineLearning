# import Pyton Packages
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#read the data set
headernames = ['pregnant', 'glucose', 'pressure', 'skin', 'insulin', 'BMI', 'pedigree', 'age', 'label']
path = r'..\csv_data\pima-indians-diabetes.csv'
dataframe = pd.read_csv (path, skiprows=9, header = None, names = headernames)
dataframe.head()
feature_cols = ['pregnant', 'insulin', 'BMI', 'age','glucose','pressure','pedigree']
X = dataframe[feature_cols]     # Features
y = dataframe.label             # Target variable

# SPlit data into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

# Prediction
y_pred = clf.predict(X_test)

# Print the Accuracy Score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)