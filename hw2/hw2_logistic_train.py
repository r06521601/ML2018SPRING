
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from math import log, floor
from random import shuffle
import os, sys

def shuf(X_train, Y_train):
    random = np.arange(len(X_train))
    np.random.shuffle(random)
    return (X_train[random], Y_train[random])

def sigmoid(z):
    n = 1 / (1.0 + np.exp(-z))
    return np.clip(n, 1e-8, 1-(1e-8))
    
def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid AC = %f' % (float(result.sum()) / valid_data_size))

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

    alldata_size = len(X_train)
    validdataset_size = int(floor(alldata_size * 0.1))# 10%

    X_all, Y_all = shuf(X_train, y_train)

    X_train, Y_train = X_all[0:validdataset_size], Y_all[0:validdataset_size]
    X_valid, Y_valid = X_all[validdataset_size:], Y_all[validdataset_size:]






    w = np.zeros((39,))
    b = np.zeros((1,))
    eta = 0.1

    train_data_size = len(X_train)
    batch_size = 32
    step_num = int(floor(train_data_size / batch_size))
    epoch_num = 1000
    save_param_iter = 50

    total_loss = 0.0
    for epoch in range(1, epoch_num):
        if (epoch) % save_param_iter == 0:
            print('save parameters for epoch %d' % epoch)
            if not os.path.exists('parameter'):
                os.mkdir('parameter')
            np.savetxt(os.path.join('parameter', 'w'), w)
            np.savetxt(os.path.join('parameter', 'b'), [b,])
            print('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
            total_loss = 0.0
            valid(w, b, X_valid, Y_valid)

        
        X_train, Y_train = shuf(X_train, Y_train)

        
        for idx in range(step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            z = np.dot(X, np.transpose(w)) + b
            y = sigmoid(z)

            cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
            total_loss += cross_entropy

            w_g = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size,1)), axis=0)
            b_g = np.mean(-1 * (np.squeeze(Y) - y))

            
            w = w - eta * w_g
            b = b - eta * b_g

if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))