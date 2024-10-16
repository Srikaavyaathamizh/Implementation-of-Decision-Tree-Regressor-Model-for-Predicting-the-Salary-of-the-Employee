# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the salary dataset and preprocess it by encoding categorical variables. 2.Split the data into training and testing sets. 3.Train a Decision Tree Regressor and predict salaries. 4.Evaluate the model using MSE and R² score.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SRIKAAVYAA T
RegisterNumber:  212223230214
*/
```

```
import pandas as pd

data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data["Position"]=le.fit_transform (data["Position"])

data.head()

x=data[["Position", "Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(x_train, y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

```
## Output:
![Screenshot 2024-10-16 154724](https://github.com/user-attachments/assets/8cd2dc19-7ad5-49d4-912a-8e34c25dc084)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
