
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sy
import pandas as pd


# In[2]:


data = []

for i in range(18):
    data.append([])

text = open('train_o.csv', 'r', encoding='big5')
#row = pd.read_csv(text,delimiter=",", skiprows=1)
row = pd.read_csv(text)

row = row.drop(row.columns[0:3], axis=1)



for i in range(0,4320):
    
    for s in range(0,24):
        if row.iloc[i][s] != "NR":
            data[(i)%18].append(float(row.iloc[i][s]))

        else:
            data[(i)%18].append(float(0))
        
    

text.close()


# In[3]:


x = []
y = []
x_1 = []
# 每 12 個月
for i in range(12):
    # 一個月取連續2小時的data可以有479筆
    for j in range(479):
        x.append([])
        x_1.append([])
        # 18種污染物
        for t in range(18):
            # 連續1小時
            for s in range(1):
                x[479*i+j].append(data[t][480*i+j+s] )
                if t == 9:
                    x_1[479*i+j].append(data[t][480*i+j+s] )
        
        
for i in range(12):
    # 一個月取連續2小時的data可以有479筆
    for j in range(479):
             
            
        y.append(data[9][480*i+j+1])
        
x_1 = np.array(x_1)
x = np.array(x)

y = np.array(y)




x_1 = np.concatenate((np.ones((x_1.shape[0],1)),x_1), axis=1)


# In[4]:


test_x = []


text = open('test.csv', 'r', encoding='big5')

row = pd.read_csv(text,header=None)


row = row.drop(row.columns[0:2], axis=1)

n=0

for i in range(0,row.shape[0]):
    if i %18 == 0:
        test_x.append([])
        
        for s in range(10,11):
            if row.iloc[i][s] != "NR":
                test_x[n//18].append(float(row.iloc[i][s]))

            else:
                test_x[n//18].append(float(0))
    else:
        
        for s in range(10,11):
            if row.iloc[i][s] != "NR":
                test_x[n//18].append(float(row.iloc[i][s]))
            ###
            else:
                test_x[n//18].append(float(0))    
    n = n+1
    
text.close()
test_x = np.array(test_x)





# In[5]:


def normalize(X):
    
    m, n = X.shape
    
    for j in range(n):
        features = X[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
           X[:,j] = (features-minVal)/diff
        else:
           X[:,j] = 0
    return X


# In[6]:


def standardize(X):
    
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X


# In[7]:


def normalize_t(X,T):
    
    m, n = X.shape
    
    for j in range(n):
        features = X[:,j]
        features_t = T[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
           T[:,j] = (features_t-minVal)/diff
        else:
           T[:,j] = 0
    return T


# In[8]:


def standardize_t(X,T):
    
    m, n = X.shape
    
    for j in range(n):
        features = X[:,j]
        features_t = T[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            T[:,j] = (features_t-meanVal)/std
        else:
            T[:, j] = 0
    return T


# In[9]:


#test_x = standardize_t(x,test_x)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)


# In[10]:


#x = standardize(x)
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


# In[11]:


w = np.zeros(len(x[0]))
learning_rate = 10
repeat = 50000
l_lambda = 0.001


# In[12]:


x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    
    #cost = (np.sum(loss**2)+np.sum(w**2)*l_lambda) /2/ len(x)
    cost = (np.sum(loss**2)) / len(x)
    cost_a  = np.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - learning_rate * gra/ada
    #w = (w*(1-(learning_rate*l_lambda)/ada))-learning_rate*gra/(0.000001+ada)
    if i %1000 ==0:
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
        print('gra:'+str(gra[0]))


# In[13]:


# save model
np.save('model.npy',w)
# read model
w = np.load('model.npy')


# In[14]:


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "result/ans.csv"
text = open(filename, "w+")


df = pd.DataFrame(ans, columns=['id', 'value'])
df.to_csv(text, sep=',',index = False)
text.close()


# In[15]:



