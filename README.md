# pk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_excel(r"C:\Users\Downloads\sample data.xlsx")
df.head()

plt.scatter(df['Age'], df['Balance'], color='red')
plt.title('Balance Vs Age', fontsize=14)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Balance', fontsize=14)
plt.grid(True)
plt.show()
 
plt.scatter(df['Education'], df['Balance'], color='green')
plt.title('Balance Vs Education', fontsize=14)
plt.xlabel('Education', fontsize=14)
plt.ylabel('Balance', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Income'], df['Balance'], color='blue')
plt.title('Balance Vs Income', fontsize=14)
plt.xlabel('Income', fontsize=14)
plt.ylabel('Balance', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Home Value'], df['Balance'], color='yellow')
plt.title('Balance Vs HomeValue', fontsize=14)
plt.xlabel('HomeValue', fontsize=14)
plt.ylabel('Balance', fontsize=14)
plt.grid(True)
plt.show()

plt.scatter(df['Wealth'], df['Balance'], color='pink')
plt.title('Balance Vs Wealth', fontsize=14)
plt.xlabel('Wealth', fontsize=14)
plt.ylabel('Balance', fontsize=14)
plt.grid(True)
plt.show()

#data_cleaning
df.isnull().any()
df = df.fillna(method='ffill')                                                        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

X = df[['Age','Education','Income','Home Value','Wealth']] 
Y = df['Balance']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

#statistical Model summary
import statsmodels.api as sm

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)



#visualizations
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import warnings
warnings. filterwarnings('ignore')
sns.boxplot(df['Balance'],orient='vertical')

sns.catplot(x='Income', y ='Balance' , kind ='violin' , data=df[:200])



