import numpy as np
import pandas as pm 
import statsmodels.api as sm
import matplotlib.pyplot as plt
 

data = pm.read_csv('D:\Python\Data_Science\Simple_linear_regression.csv')
data.describe()

y = data['GPA']
x1 = data['SAT']

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

plt.scatter(x1,y)
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1,yhat, lw=4, c='orange',label='Regression Line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()
