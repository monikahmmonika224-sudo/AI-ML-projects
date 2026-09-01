import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r'mtcars1.csv')
print(df.head(10))
print(plt.hist(x=df['mpg']))
plt.show()
print(plt.scatter(x='wt',y='mpg',data=df))
plt.show()
print(df['am'].value_counts().plot(kind='bar'))
plt.show()
print(sns.boxplot(df['mpg']))
plt.show()
print(df['mpg'].min())
print(df['mpg'].max())
print(df['mpg'].quantile([.1,.25,.5,.25]))

