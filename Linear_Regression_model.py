import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Dataset = pd.read_csv(r"D:\VS_CODE\ML_Practical\Salary_Data.csv")
print("Dataset Shape:", Dataset.shape)

x = Dataset.iloc[:, :-1]
y = Dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)



y_pred = regressor.predict(x_test)
print(y_pred)


comparision = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
print(comparision)


plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

model_coef = regressor.coef_
print(model_coef)

model_const = regressor.intercept_
print(model_const)

y_12 = model_coef*12+model_const
print(y_12)

y_20 = model_coef*20+model_const
print(y_20)

# Mean
Dataset.mean()   # this will give mean of entire dataframe
Dataset['Salary'].mean() # this will give us mean of that particular column


# median
Dataset.median() # this will give median of entire dataframe
Dataset['Salary'].median()  # this will give us median of thay particular column

#Mode
Dataset['Salary'].mode() # This will give mode of that particular column


# Variance
Dataset.var()  # This will give variance of entire dataframe
Dataset['Salary'].var()  # This will give us variance of that particular columns

# Standard Deviation
Dataset.std()  #  Tis will give Standard Deviation of entire Dataframe
Dataset['Salary'].std()  # this will give us standard deviation of that particular column

# Coefficient of Variation
from scipy.stats import variation
variation(Dataset.values)  # This will give CV of entire dataframe
variation(Dataset['Salary'])  # this will give us CV of that particular columns


# Correlation 
Dataset.corr()  # this will give correlation of entire Dataframe
Dataset['Salary'].corr(Dataset['YearsExperience'])# this will give us correlation between these


# Skewness
Dataset.skew() # this will give skewness od entire datafram
Dataset['Salary'].skew()
 
 # Standard Error
Dataset.sem()  # this will give standard error of entire dataframe
Dataset['Salary'].sem() # this will give us Standard error of that particular column


# Z-Score

import scipy.stats as stats
Dataset.apply(stats.zscore) # this will give Z-score od entire dataframe 
stats.zscore(Dataset['Salary'])  # this will give us Z-Score of that particular columns


# Sum of Squares Regression (SSR) 
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)


# Sum of Squares Error (SSE)
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# Sum of Squares Total (SST)
mean_total = np.mean(Dataset.values)
SST = np.sum((Dataset.values-mean_total)**2)
print(SST)

# R_Square
r_square = 1 - SSR/SST
print(r_square)

# Bias_Square
bias_score = regressor.score(x_train, y_train)
print(bias_score)


# Variance_Score
variance = regressor.score(x_test, y_test)
print(variance)

import pickle

filename = 'liner_regression_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor, file)
    
print("Model has been Pickled and saved as linear_regression_model.pkl")

import os
os.getcwd()