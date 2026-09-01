import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\lenovo\Downloads\housing.csv")
df=pd.get_dummies(df)
x=df.iloc[:,:-1:]
y=df.iloc[:, -1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
y_pred=lm.predict(x_test)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("MSE --->",mean_squared_error(y_test,y_pred))
print("RMSE --->",np.sqrt(mean_squared_error(y_test,y_pred)))
print("MAE --->",mean_absolute_error(y_test,y_pred))
print("r2 score --->",r2_score(y_test,y_pred))