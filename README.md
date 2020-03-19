# pk

#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
 
data = pd.read_excel(r'Documents\\sample.xlsx')
print(data.head(5))


# In[13]:


plt.figure(figsize=(16, 8))
plt.scatter(
    data['Income'],
    data['Balance'],
    c='black'
)
plt.xlabel("Income")
plt.ylabel("Balance")
plt.show()


# In[20]:


Xs = data.drop(['Balance'], axis=1)
y = data['Balance']
reg = LinearRegression()
reg.fit(Xs, y)
#print("The linear model is: Y = {:.5} + {:.5}*Age + {:.5}*Education + {:.5}*Income + {:.5}*Home Value + {:.5}*Wealth".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2], reg.coef_[0][3], reg.coef_[0][4]))


# In[21]:


X = np.column_stack((data['Age'], data['Education'], data['Income'], data['Home Value'], data['Wealth']))
y = data['Balance']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:




