import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
df=pd.read_csv(r"C:\Users\lenovo\Downloads\creditcard.csv")
print(df.head())
print(df.isna().sum())
for col in['nameOrig' and 'nameDest']:
    if col in df.columns:
        del df[col]
if'isFraud' not in df.columns and 'Class' in df.columns:
    df.rename(columns={'Class':'isFraud'},inplace=True)
if'step' not in df.columns:
    df['step']=(df['Time']%(24*3600))//3600+1
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
df['isFraud'].value_counts().plot(kind='pie',autopct='%1.0f%%')
plt.ylabel('isFraud')
plt.show()
df['step']=df['step']%24+1
df=df.sort_values(by='step')
plt.subplot(1,2,2)
df['step'].value_counts().sort_index().plot.pie(autopct='%1.2f%%')
plt.ylabel('step')
plt.tight_layout()
plt.show()
if'type' in df.columns:
    le=LabelEncoder()
    df['type']=le.fit_transform(df['type'])
print(df.head())
x=df.drop(columns=['isFraud'])
y=df['isFraud']
sm=SMOTE(random_state=42)
x_sm,y_sm=sm.fit_resample(x,y)
print(f"Shape of X before SMOTE:{x.shape}")
print(f"Shape of X after SMOTE:{x_sm.shape}")
print('\nBalance of positive and negative classes(%):')
print(y_sm.value_counts(normalize=True)*100)
x_train,x_test,y_train,y_test=train_test_split(x_sm,y_sm,test_size=0.2,random_state=42)
rm=RandomForestClassifier(random_state=42)
rm.fit(x_train,y_train)
y_pred=rm.predict(x_test)
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test,y_pred))
print("ClassificationReport:\n",classification_report(y_test,y_pred))