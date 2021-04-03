#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
os.getcwd()
os.chdir(r'C:\Users\Rashmi\Desktop\Project-2\Project-2')


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report


# In[42]:


df=pd.read_csv('heart_disease_patients.csv')


# In[43]:


df


# In[44]:


df


# In[45]:


df.dtypes


# In[46]:


df.isnull().sum()


# In[47]:


df.corr()


# In[48]:


df = df[['age','cp', 'trestbps', 'exang', 'oldpeak', 'slope']]
df


# In[49]:


x=df.iloc[:,1:6].values


# In[50]:


x


# In[51]:


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x=sc_x.fit_transform(x)
x


# In[52]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[53]:


wcss


# In[54]:


plt.plot(range(1,11),wcss)


# In[20]:


sns.lmplot(x='age', y='cp', data= df)


# In[270]:


sns.lmplot(x='age', y='trestbps', data= df)


# In[272]:


sns.lmplot(x='age', y='exang', data= df,fit_reg=False)


# In[273]:


sns.lmplot(x='age', y='oldpeak', data= df,fit_reg=False)


# In[274]:


sns.lmplot(x='age', y='slope', data= df,fit_reg=False)


# In[275]:


sns.lmplot(x='cp', y='trestbps', data= df,fit_reg=False)


# In[276]:


sns.lmplot(x='cp', y='exang', data= df,fit_reg=False)


# In[277]:


sns.lmplot(x='cp', y='exang', data= df,fit_reg=False)


# In[278]:


sns.lmplot(x='cp', y='oldpeak', data= df,fit_reg=False)


# In[279]:


sns.lmplot(x='cp', y='slope', data= df,fit_reg=False)


# In[280]:


sns.lmplot(x='trestbps', y='exang', data= df,fit_reg=False)


# In[281]:


sns.lmplot(x='trestbps', y='oldpeak', data= df,fit_reg=False)


# In[282]:


sns.lmplot(x='trestbps', y='slope', data= df,fit_reg=False)


# In[283]:


sns.lmplot(x='exang', y='oldpeak', data= df,fit_reg=False)


# In[284]:


sns.lmplot(x='exang', y='slope', data= df,fit_reg=False)


# In[285]:


sns.lmplot(x='oldpeak', y='slope', data= df,fit_reg=False)


# In[55]:


kmeans = KMeans(n_clusters=19, init='k-means++')
y_kmeans = kmeans.fit_predict(x)


# In[56]:


y_kmeans


# In[57]:


df


# In[58]:


df=pd.concat([df,pd.DataFrame(y_kmeans)], axis=1)


# In[59]:


df


# In[60]:


from sklearn.metrics import silhouette_score
print(silhouette_score(x,y_kmeans))


# In[61]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))


# In[62]:


from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(x)


# In[63]:


y_hc


# In[64]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=50,c='red')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=50,c='green')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=50,c='blue')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=50,c='black')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=50,c='yellow')


# In[ ]:




