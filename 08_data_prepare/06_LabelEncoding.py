import numpy as np
from sklearn import preprocessing
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit (input_labels)
encoded_input_values = encoder.transform (input_labels)
print ("\ninput_labels =", input_labels)
print ("\nencoded_input_values =", encoded_input_values)
test_labels = ['green', 'red', 'black']

encoded_values = encoder.transform (test_labels)
print ("\nLabels =", test_labels)
print ("\nencoded_values =", encoded_values)
encoded_values = [3, 0, 4, 1]
encoded_list = encoder.inverse_transform (encoded_values)
print("\nencoded_values =", encoded_values)
print("\nencoded_list =", encoded_list)