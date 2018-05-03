
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import os, sys
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense


# In[2]:


df = pd.read_csv('train.csv', header=None, skiprows=1)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
df.isnull().values.any()
df.Income.unique()
df["Income"] = df["Income"].map({ " <=50K": 0, " >50K": 1 })
y_all = df["Income"].values
df.drop("Income", axis=1, inplace=True,)
df.drop("Relationship", axis=1, inplace=True,)
df.drop("NativeCountry", axis=1, inplace=True,)
df.drop("Race", axis=1, inplace=True,)
df.drop("Education", axis=1, inplace=True,)
df.Age = df.Age.astype(float)
df.fnlwgt = df.fnlwgt.astype(float)
df.EducationNum = df.EducationNum.astype(float)
df.HoursPerWeek = df.HoursPerWeek.astype(float)
df = pd.get_dummies(df, columns=[
    "WorkClass",  "MaritalStatus", "Occupation", 
     "Gender", 
])
df.Age = (df.Age-df.Age.mean())/df.Age.std()
df.fnlwgt = (df.fnlwgt-df.fnlwgt.mean())/df.fnlwgt.std()
df.EducationNum = (df.EducationNum-df.EducationNum.mean())/df.EducationNum.std()
df.HoursPerWeek = (df.HoursPerWeek-df.HoursPerWeek.mean())/df.HoursPerWeek.std()
df.CapitalGain = (df.CapitalGain-df.CapitalGain.mean())/df.CapitalGain.std()
df.CapitalLoss = (df.CapitalLoss-df.CapitalLoss.mean())/df.CapitalLoss.std()

X_train = df.values
y_train = y_all


# In[3]:


model_1 = Sequential()
model_1.add(Dense(39, input_dim=39,  activation='sigmoid'))
model_1.add(Dense(1,  activation='sigmoid'))
from keras.optimizers import SGD,Adam
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy
model_1.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy, metrics=[binary_accuracy])


# In[4]:


model_1.summary()


# In[5]:


train_history = model_1.fit(X_train, y_train, epochs = 50,verbose=1, batch_size=128,validation_split=0.1)


# In[7]:


model_1.save('model_1.h5')


# In[30]:


model_2 = Sequential()
model_2.add(Dense(39, input_dim=39,  activation='sigmoid'))
model_2.add(Dense(14,  activation='relu'))
model_2.add(Dense(1,  activation='sigmoid'))
model_2.compile(optimizer=SGD(lr=0.01), loss=binary_crossentropy, metrics=[binary_accuracy])
model_2.summary()


# In[31]:


train_history = model_2.fit(X_train, y_train, epochs = 50,verbose=1, batch_size=128,validation_split=0.1)


# In[32]:


model_2.save('model_2.h5')


# In[33]:


from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization


# In[34]:


model_3 = Sequential()
model_3.add(Dense(39, input_dim=39,  activation='sigmoid'))
model_3.add(Dense(14,  activation='relu'))
model_3.add(BatchNormalization())
model_3.add(Dropout(0.3))
model_3.add(Dense(1,  activation='sigmoid'))
model_3.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy, metrics=[binary_accuracy])
model_3.summary()


# In[35]:


train_history = model_3.fit(X_train, y_train, epochs = 50,verbose=1, batch_size=128,validation_split=0.1)


# In[36]:


model_3.save('model_3.h5')


# In[66]:


df = pd.read_csv('train.csv', header=None, skiprows=1)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]
df.isnull().values.any()
df.Income.unique()
df["Income"] = df["Income"].map({ " <=50K": 0, " >50K": 1 })
y_all = df["Income"].values
df.drop("Income", axis=1, inplace=True,)
df.drop("Relationship", axis=1, inplace=True,)
df.drop("NativeCountry", axis=1, inplace=True,)
df.drop("Race", axis=1, inplace=True,)
df.drop("Education", axis=1, inplace=True,)
df.Age = df.Age.astype(float)
df.fnlwgt = df.fnlwgt.astype(float)
df.EducationNum = df.EducationNum.astype(float)
df.HoursPerWeek = df.HoursPerWeek.astype(float)
df = pd.get_dummies(df, columns=[
    "WorkClass",  "MaritalStatus", "Occupation", 
     "Gender", 
])
df1 = pd.read_csv('test.csv', header=None, skiprows=1)
df1.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry"
]
df1.isnull().values.any()
df1.drop("Relationship", axis=1, inplace=True,)
df1.drop("NativeCountry", axis=1, inplace=True,)
df1.drop("Race", axis=1, inplace=True,)
df1.drop("Education", axis=1, inplace=True,)
df1.Age = df1.Age.astype(float)
df1.fnlwgt = df1.fnlwgt.astype(float)
df1.EducationNum = df1.EducationNum.astype(float)
df1.HoursPerWeek = df1.HoursPerWeek.astype(float)
df1.Age = (df1.Age-df.Age.mean())/df.Age.std()
df1.fnlwgt = (df1.fnlwgt-df.fnlwgt.mean())/df.fnlwgt.std()
df1.EducationNum = (df1.EducationNum-df.EducationNum.mean())/df.EducationNum.std()
df1.HoursPerWeek = (df1.HoursPerWeek-df.HoursPerWeek.mean())/df.HoursPerWeek.std()
df1.CapitalGain = (df1.CapitalGain-df1.CapitalGain.mean())/df1.CapitalGain.std()
df1.CapitalLoss = (df1.CapitalLoss-df1.CapitalLoss.mean())/df1.CapitalLoss.std()
df.Age = (df.Age-df.Age.mean())/df.Age.std()
df.fnlwgt = (df.fnlwgt-df.fnlwgt.mean())/df.fnlwgt.std()
df.EducationNum = (df.EducationNum-df.EducationNum.mean())/df.EducationNum.std()
df.HoursPerWeek = (df.HoursPerWeek-df.HoursPerWeek.mean())/df.HoursPerWeek.std()
df.CapitalGain = (df.CapitalGain-df.CapitalGain.mean())/df.CapitalGain.std()
df.CapitalLoss = (df.CapitalLoss-df.CapitalLoss.mean())/df.CapitalLoss.std()
df1 = pd.get_dummies(df1, columns=[
    "WorkClass", "MaritalStatus", "Occupation",
     "Gender",
])

missing_cols = set( df.columns ) - set( df1.columns )
for c in missing_cols:
    df1[c] = 0
df1 = df1[df.columns]
X_train = df.values
y_train = y_all
X_test = df1.values


# In[67]:


predictions_1 = model_1.predict(X_test)
predictions_2 = model_2.predict(X_test)
predictions_3 = model_3.predict(X_test)


# In[69]:


prediction_list = np.array([predictions_1,predictions_2,predictions_3])


# In[74]:


prediction_ensemble = np.average(prediction_list, axis=0, weights=[0.8,1,1])


# In[75]:



rounded = [round(x[0]) for x in prediction_ensemble]
map(int, rounded)



filename = 'ans.csv'
if not os.path.exists(filename):
    os.mkdir(filename)
output_path = os.path.join(filename, 'prediction.csv')

ans = []
for i in range(len(X_test)):
    ans.append([str(i+1)])
    a = int(rounded[i])
    ans[i].append(a)

text = open(output_path, "w+")


df = pd.DataFrame(ans, columns=['id', 'label'])
df.to_csv(text, sep=',',index = False)
text.close()

