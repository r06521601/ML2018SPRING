
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
    df = pd.read_csv(argv[0], header=None, skiprows=1)


    # In[3]:


    df.columns = [
            "label", "feature"]


    # In[4]:


    


    # In[5]:


    y_all = df["label"].values


    # In[7]:


    df.drop("label", axis=1, inplace=True,)


    # In[8]:


    y_trainonehot = np_utils.to_categorical(y_all)


    # In[9]:


    x_train_col = df['feature']


    # In[10]:


    x_train = []


    # In[11]:


    for i in range(len(x_train_col)):
        x_train_col[i] = np.fromstring(x_train_col[i],dtype=int,sep=' ')


    # In[12]:


    for i in range(len(x_train_col)):
        x_train.append(np.reshape(x_train_col[i],(48,48,1)))


    # In[13]:


    


    # In[14]:


    x_train = np.array(x_train)/255
    

    #x_train = (x_train-x_train.mean())/x_train.std()


    # In[15]:


    x_train_pre = x_train


    # In[16]:


    from keras import regularizers
    from keras.layers.normalization import BatchNormalization


    # In[17]:



    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(

        zca_whitening=False,

        rotation_range=30,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')




    # In[18]:


    model = Sequential()
    
    model.add(Conv2D(filters = 64,
                    kernel_size=(3,3),
                    padding='same',                
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64,
                    kernel_size=(3,3),
                    padding='same',
                    activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters = 128,
                    kernel_size=(3,3),
                    padding='same',
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128,
                    kernel_size=(3,3),
                    padding='same',
                    activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(7, activation='softmax'))

    from keras.optimizers import SGD,Adam, RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00123),metrics=['accuracy'])
    model.summary()


    # In[56]:


    datagen.fit(x_train_pre)

    train_history = model.fit_generator(datagen.flow(x_train_pre,  

          y_trainonehot, batch_size=128),

          steps_per_epoch=round(len(x_train_pre)/128),

                epochs=500, validation_data=(x_train_pre, y_trainonehot))


    # In[27]:


    from keras.models import load_model


    # In[57]:


    model.save("my_model.h5")
    del model

if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))