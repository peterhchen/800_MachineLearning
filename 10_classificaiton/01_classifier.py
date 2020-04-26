# Step 1: import python package
import sklearn
# Step 2: import data set
from sklearn.datasets import load_breast_cancer

breast_data = load_breast_cancer()
print ('breast_data:\n', breast_data) 
# the breast_data content are as follow:
# breast_data:
# {'data': array(...), 
# 'target': array(...),
# 'target_names': array (...), 
# 'DESCR': '...', 
# 'feature_names': array (...),
# 'filename': 'C:\\Users\\14088\\AppData\\Local\\Programs\\Python
# \\Python38-32\\lib\\site-packages\\sklearn\\datasets\\data\\breast_cancer.csv'
# }
# Note:
# after load_breast_cancer(), the data 
# You csv data from the above path
data = breast_data['data']
target = breast_data['target']
target_names = breast_data['target_names']
DESCR = breast_data['DESCR']
feature_names = breast_data['feature_names']
filename=breast_data['filename']

print ('\n\ndata:\n', data [:2])
# data:
#  [[1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
#   1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
#   6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
#   1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
#   4.601e-01 1.189e-01]
#  [2.057e+01 1.777e+01 1.329e+02 1.326e+03 8.474e-02 7.864e-02 8.690e-02
#   7.017e-02 1.812e-01 5.667e-02 5.435e-01 7.339e-01 3.398e+00 7.408e+01
#   5.225e-03 1.308e-02 1.860e-02 1.340e-02 1.389e-02 3.532e-03 2.499e+01
#   2.341e+01 1.588e+02 1.956e+03 1.238e-01 1.866e-01 2.416e-01 1.860e-01
#   2.750e-01 8.902e-02]]
print ('\ntarget:\n', target[:569])
# The series of 0s and 1s in output are the predicted values for the 
# Malignant (0) and Benign (1) tumor classes.
# target:
#   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
#  1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
#  1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
#  1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
#  1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
#  1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
#  1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
#  1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
#  0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
#  1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
#  0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
#  1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
#  1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 0 0 0 0 0 0 1]
# 212: malignant (label = 0), 357: benign (label = 1)
print ('\ntarget_names:', target_names[:20])
# target_names: ['malignant' 'benign']
print ('\nDESCR:\n', DESCR[:5000])
# DESCR:
#  .. _breast_cancer_dataset:

# Breast cancer wisconsin (diagnostic) dataset
# --------------------------------------------

# **Data Set Characteristics:**

#     :Number of Instances: 569

#     :Number of Attributes: 30 numeric, predictive attributes and the class

#     :Attribute Information:
#         - radius (mean of distances from center to points on the perimeter)
#         - texture (standard deviation of gray-scale values)
#         - perimeter
#         - area
#         - smoothness (local variation in radius lengths)
#         - compactness (perimeter^2 / area - 1.0)
#         - concavity (severity of concave portions of the contour)
#         - concave points (number of concave portions of the contour)
#         - symmetry
#         - fractal dimension ("coastline approximation" - 1)

#         The mean, standard error, and "worst" or largest (mean of the three
#         largest values) of these features were computed for each image,
#         resulting in 30 features.  For instance, field 3 is Mean Radius, field
#         13 is Radius SE, field 23 is Worst Radius.

#         - class:
#                 - WDBC-Malignant
#                 - WDBC-Benign

#     :Summary Statistics:

#     ===================================== ====== ======
#                                            Min    Max
#     ===================================== ====== ======
#     radius (mean):                        6.981  28.11
#     texture (mean):                       9.71   39.28
#     perimeter (mean):                     43.79  188.5
#     area (mean):                          143.5  2501.0
#     smoothness (mean):                    0.053  0.163
#     compactness (mean):                   0.019  0.345
#     concavity (mean):                     0.0    0.427
#     concave points (mean):                0.0    0.201
#     symmetry (mean):                      0.106  0.304
#     fractal dimension (mean):             0.05   0.097
#     radius (standard error):              0.112  2.873
#     texture (standard error):             0.36   4.885
#     perimeter (standard error):           0.757  21.98
#     area (standard error):                6.802  542.2
#     smoothness (standard error):          0.002  0.031
#     compactness (standard error):         0.002  0.135
#     concavity (standard error):           0.0    0.396
#     concave points (standard error):      0.0    0.053
#     symmetry (standard error):            0.008  0.079
#     fractal dimension (standard error):   0.001  0.03
#     radius (worst):                       7.93   36.04
#     texture (worst):                      12.02  49.54
#     perimeter (worst):                    50.41  251.2
#     area (worst):                         185.2  4254.0
#     smoothness (worst):                   0.071  0.223
#     compactness (worst):                  0.027  1.058
#     concavity (worst):                    0.0    1.252
#     concave points (worst):               0.0    0.291
#     symmetry (worst):                     0.156  0.664
#     fractal dimension (worst):            0.055  0.208
#     ===================================== ====== ======

#     :Missing Attribute Values: None

#     :Class Distribution: 212 - Malignant, 357 - Benign

#     :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian

#     :Donor: Nick Street

#     :Date: November, 1995

# This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
# https://goo.gl/U2Uwz2

# Features are computed from a digitized image of a fine needle
# aspirate (FNA) of a breast mass.  They describe
# characteristics of the cell nuclei present in the image.

# Separating plane described above was obtained using
# Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
# Construction Via Linear Programming." Proceedings of the 4th
# Midwest Artificial Intelligence and Cognitive Science Society,
# pp. 97-101, 1992], a classification method which uses linear
# programming to construct a decision tree.  Relevant features
# were selected using an exhaustive search in the space of 1-4
# features and 1-3 separating planes.

# The actual linear program used to obtain the separating plane
# in the 3-dimensional space is that described in:
# [K. P. Bennett and O. L. Mangasarian: "Robust Linear
# Programming Discrimination of Two Linearly Inseparable Sets",
# Optimization Methods and Software 1, 1992, 23-34].

# This database is also available through the UW CS ftp server:

# ftp ftp.cs.wisc.edu
# cd math-prog/cpo-dataset/machine-learn/WDBC/

# .. topic:: References

#    - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction
#      for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on
#      Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
#      San Jose, CA, 1993.
#    - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and
#      prognosis via linear programming. Operations Research, 43(4), pages 570-577,
#      July-August 1995.
#    - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
#      to diagnose breast cancer from fine-needle aspirates.
print ('\nfeature_names:\n', feature_names[:50])
# feature_names:
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
# 'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
print ('\nfilename:\n', filename[:150])
# filename:
# C:\Users\14088\AppData\Local\Programs\Python\Python38-32\lib\site-packages\
# sklearn\datasets\data\breast_cancer.csv

# Step 3: Organize data into training and testing sets
# features => data
# labels => target
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = \
    train_test_split (data, target, test_size=0.40, random_state=42)

# Step 4: Model Selection
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print ('\npredictions:\n', preds)

# Step 5: Finding Accuracy
from sklearn.metrics import accuracy_score
print ('\naccuracy_score:\n', accuracy_score (test_labels, preds))
