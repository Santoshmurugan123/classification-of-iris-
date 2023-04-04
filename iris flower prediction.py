#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris=pd.read_csv("C:\\Users\\SANTH\\OneDrive\\Documents\\IRIS.csv")
iris
x = iris.drop("Species", axis=1)
y = iris["Species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = np.array([[5,5,3.4,1.5,0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[ ]:




