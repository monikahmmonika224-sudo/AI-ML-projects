import numpy as np
import pandas as pd
import matlotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv(r"athlete_events.csv")
print(data.head(10))
data['Height']=data['Height'].fillna(data['Height'].mean())
data['Weight']=data['Weight'].fillna(data['Weight'].mean())
print(datainfo())
print(data['Weight'].skew())
plt.hist(data['Weight'])
plt.show()
sns.boxplot(data['Weight'])
plt.show
q1=data['Weight'].quantile(0.25)
q3=data['Weight'].quantile(0.75)
IQR=q3-q1
lower=q1-(1.5*