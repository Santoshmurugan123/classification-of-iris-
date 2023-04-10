#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris=pd.read_csv("C:\\Users\\SANTH\\OneDrive\\Documents\\Iris.csv")
iris


# In[29]:


print(iris.head())


# In[30]:


iris.describe()


# # distribution of data

# In[25]:


import plotly.express as px
fig = px.scatter(iris, x="SepalLengthCm", y="PetalLengthCm", color="Species")
fig.show()


# # declaring two variables

# In[35]:


x = iris.drop("Species", axis=1)
y = iris["Species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# # using classification algorithm

# In[36]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = np.array([[6.3,2.5,5.0,1.9]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[ ]:




