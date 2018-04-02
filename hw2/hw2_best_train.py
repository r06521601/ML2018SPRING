
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd 
import tensorflow as tf
import os, sys
from keras.models import Sequential
from keras.models import load_model

from keras.layers.core import Dense


# In[2]:

def main(argv):  

    df = pd.read_csv(argv[0], header=None, skiprows=1)
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

    model = Sequential()
    model.add(Dense(39, input_dim=39,  activation='sigmoid'))
    model.add(Dense(1,  activation='sigmoid'))
    from keras.optimizers import SGD,Adam
    from keras.objectives import binary_crossentropy
    from keras.metrics import binary_accuracy
    model.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy, metrics=[binary_accuracy])




    train_history = model.fit(X_train, y_train, epochs = 50,verbose=1, batch_size=64,validation_split=0.1)





    model.save('my_model.h5')
    del model


if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))



