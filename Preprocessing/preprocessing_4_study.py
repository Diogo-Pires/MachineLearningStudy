#All rights to the Udemy course: Learn to create Machine Learning Algorithms in Python and R from two Data Science experts. Code templates included.
#https://ust-global.udemy.com/course/machinelearning/learn/lecture/19697076#overview
#I'm studying machine learning using the above mentioned course. 

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
dataset = pd.read_csv('iris.csv')

# Separate features and target
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature scaling on the training and test sets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the scaled training and test sets
print("Scaled Training Set:")
print(X_train)
print("\nScaled Test Set:")
print(X_test)