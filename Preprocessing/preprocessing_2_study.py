#All rights to the Udemy course: Learn to create Machine Learning Algorithms in Python and R from two Data Science experts. Code templates included.
#https://ust-global.udemy.com/course/machinelearning/learn/lecture/19697076#overview
#I'm studying machine learning using the above mentioned course. 

# Importing the necessary libraries
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Load the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Identify missing data (assumes that missing data is represented as NaN)
rows_with_nan = np.where(np.isnan(x))[0]  # Indices of rows with NaNs
cols_with_nan = np.where(np.isnan(x))[1]  # Indices of columns with NaNs

# Print the number of missing entries in each column
print("Rows with NaN values:", np.unique(rows_with_nan))
print("Columns with NaN values:", np.unique(cols_with_nan))

#Print first row where we have a NAN value
print(x[0])

df = pd.DataFrame(x)
print(df.isnull().sum()) 

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(x[:, 1:9])

# Apply the transform to the DataFrame
x[:, 1:9] = imputer.transform(x[:, 1:9])

#Print first row where we have a NAN value
print(x[0])

#Print your updated matrix of features
print(x)