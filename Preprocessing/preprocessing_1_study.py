#All rights to the Udemy course: Learn to create Machine Learning Algorithms in Python and R from two Data Science experts. Code templates included.
#https://ust-global.udemy.com/course/machinelearning/learn/lecture/19697076#overview
#I'm studying machine learning using the above mentioned course. 

# Importing the necessary libraries
import numpy as np
import pandas as pd
import sklearn.model_selection as train_test_split 

# Loading the Iris dataset
dataset = pd.read_csv('iris.csv')

# Creating thetrix of features (X) and the dependent variable vector (y)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Printing the matrix of features and the dependent variable vector
print(x)
print(y)