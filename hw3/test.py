
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
import os
import sys

os.environ["THEANO_FLAGS"] = "device=gpu0"
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D


# In[2]:

def main(argv):



    # In[4]:


    df1 = pd.read_csv(argv[0], header=None, skiprows=1)
    df1.columns = ["id", "feature"]

    x_test = []    
    df1.drop("id", axis=1, inplace=True,)
    x_test_col = df1['feature']



    # In[13]:


    for i in range(len(x_test_col)):
        x_test_col[i] = np.fromstring(x_test_col[i],dtype=int,sep=' ')
        x_test.append(np.reshape(x_test_col[i],(48,48,1)))



    x_test = np.array(x_test)/255




    # In[27]:


    from keras.models import load_model


    # In[57]:

    model = load_model("my_model.h5")

    # In[58]:


    model_predictions = model.predict_classes(x_test)
    ans = []
    for i in range(len(x_test)):
        ans.append([str(i)])
        a = model_predictions[i]
        ans[i].append(a)




    text = open(argv[1], "w+")
    df1 = pd.DataFrame(ans, columns=['id', 'label'])
    df1.to_csv(text, sep=',',index = False)
    text.close()

if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))