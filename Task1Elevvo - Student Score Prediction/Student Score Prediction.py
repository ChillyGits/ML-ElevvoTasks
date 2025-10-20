import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error


#opening and cleaning Data
student_data = pd.read_csv("StudentPerformanceFactors.csv")
print(student_data.isnull().any())
student_data["Hours_Studied"] = student_data["Hours_Studied"].bfill()
student_data["Exam_Score"] = student_data["Exam_Score"].bfill()
xpoints = student_data.loc[:,['Hours_Studied']]
ypoints = student_data.loc[:,'Exam_Score']

#spliting data into training set and testing set
X_train,X_test,y_train,y_test = train_test_split(xpoints,ypoints,train_size = 0.8,random_state=10) #testing=0.2
LRModel = LinearRegression() #linear regression model
LRModel.fit(X_train,y_train)    #fitting data to train the model

x_sorted = X_train.sort_values(by="Hours_Studied")#sorts data to pervent jagged lines
y_pred_sorted = LRModel.predict(x_sorted)

# polynomial features (2nd degree)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

x_range = np.linspace(xpoints['Hours_Studied'].min(), xpoints['Hours_Studied'].max(), 200).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range = poly_model.predict(x_range_poly)

plt.scatter(xpoints,ypoints,color='blue',alpha = 0.2,label='Actual Scores')#actual scores
plt.plot(x_sorted, y_pred_sorted, color='red',label='linear Regression Model') #linear Regression
plt.plot(x_range, y_range, color='green',label='Polynomial Regression Model')#polynomial Regression(2nd Degree)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)#graph look
plt.title('Student Score Prediction')
plt.xlabel('Number of Hours Studied')
plt.ylabel('Final Exam Score ')
plt.legend()
plt.show()