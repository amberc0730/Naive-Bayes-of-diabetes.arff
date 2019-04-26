#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
#讀取CSV檔案
data = pd.read_csv('diabetes.csv') 


# In[16]:


data.info()


# In[17]:


data.describe ( )


# In[18]:


#x:input
x = data.loc[:,['preg','plas','pres','skin','insu','mass','pedi','age']]
#y:output
y = data.loc[:,['class']]


# In[19]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

X_preg_encoded=x.preg
X_plas_encoded=x.plas
X_pres_encoded=x.pres
X_skin_encoded=x.skin
X_insu_encoded=x.insu
X_mass_encoded=x.mass
X_pedi_encoded=x.pedi
X_age_encoded=x.age

#將class轉為數字label
#class: no: 0 ,yes: 1
Y_class_label=le.fit_transform(y['class'])

#將屬性合併
#變成list
feature=list(zip(X_preg_encoded, X_plas_encoded,X_pres_encoded,X_skin_encoded,X_insu_encoded, X_mass_encoded,X_pedi_encoded,X_age_encoded))

#轉成array
import numpy as np
features=np.asarray(feature)


# In[20]:


#Import Gaussian Naive Bayes 模型 (高斯樸素貝氏)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

# 訓練集訓練模型
# model.fit(x, y)
model.fit(features, Y_class_label)


# In[21]:


expected = Y_class_label
predicted = model.predict(features)
from sklearn import metrics
print(metrics.classification_report(expected, predicted))


# In[22]:


print(metrics.confusion_matrix(expected, predicted))


# In[23]:


predicted = model.predict([[2,1,0,0,2,1,2,20]])
print ("Predicted Value:", predicted)


# In[ ]:





# In[ ]:





# In[ ]:




