#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df = pd.read_csv('Z:\diabetes.csv')
df.head()


# In[35]:


#summary
df.info()


# In[36]:


#THE MEAN, VARIANCE AND STANDARD DEVIATION FOR AGE
mean = df['Pregnancies'].mean()
var = df['Pregnancies'].var()
std = df['Pregnancies'].std()
print (f"Mean {mean}\nVariance {var}\nStandard Deviation = {std}")


# In[37]:


#THE MEAN, VARIANCE AND STANDARD DEVIATION FOR 
mean = df['Glucose'].mean()
var = df['Glucose'].var()
std = df['Glucose'].std()
print (f"Mean {mean}\nVariance {var}\nStandard Deviation = {std}")


# In[38]:


plt.figure(figsize=(10,2))
plt.boxplot(df['BMI'], vert=False, showmeans=True) 
plt.grid(color='blue', linestyle='dotted')
plt.show()


# In[39]:


plt.figure(figsize=(10,2))
plt.boxplot(df['Insulin'], vert=False, showmeans=True) 
plt.grid(color='blue', linestyle='dotted')
plt.show()


# In[40]:


df['Age'].hist (bins=15) 
plt.suptitle('AGE distribution of diabetes patient') 
plt.xlabel('Age')
plt.ylabel('Count') 
plt.show()


# In[47]:


plt.scatter(df['BMI'],df['Insulin'])
plt.xlabel('BMI')
plt.ylabel('Insulin')
plt.show()


# In[42]:


sns.histplot(df['BloodPressure'], kde=True)
plt.title('Histogram with KDE')
plt.show()


# In[44]:


fig = px.scatter(df, x='Age', y='BloodPressure', title='Scatter Plot')
fig.show()


# In[48]:


from bokeh.plotting import figure, show
p = figure(title='Bokeh Example', x_axis_label='X-axis', y_axis_label='Y-axis')
p.circle(df['Age'], df['SkinThickness'], size=10)
show(p)


# In[ ]:




