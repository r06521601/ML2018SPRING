
	# coding: utf-8

# In[3]:


import numpy as np
import keras
import pandas as pd

import os

import sys
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam



from sklearn.cluster import KMeans

	# In[2]:

def main(argv):
	x = np.load(argv[0])
	x = x.astype('float32')/255
	x = np.reshape(x,(len(x),-1))


	# In[ ]:


	x_train = x[:13000]
	x_val = x[13000:]


	# In[3]:




	# In[ ]:


	input_img = Input(shape=(784,))


	# In[ ]:


	encoded = Dense(128, activation='relu')(input_img)
	encoded = Dense(64, activation='relu')(encoded)
	encoded = Dense(32, activation='relu')(encoded)
	decoded = Dense(64, activation='relu')(encoded)
	decoded = Dense(128, activation='relu')(decoded)
	decoded = Dense(784, activation='relu')(decoded)


	# In[ ]:


	encoder = Model(inputs=input_img, outputs=encoded)

	autoencoder = Model(inputs=input_img, outputs=decoded)

	autoencoder.compile(optimizer = Adam(lr=0.0005), loss = 'mse')


	# In[ ]:


	autoencoder.summary()


	# In[ ]:


	autoencoder.fit(x_train, x_train,
				   epochs=50,
				   batch_size=256,
				   shuffle=True,
				   validation_data=(x_val,x_val))
	autoencoder.save('autoencoder.h5')
	encoder.save('encoder.h5')


	# In[1]:



	# In[5]:
	encoded_img = encoder.predict(x)


	# In[6]:


	encoded_img = encoded_img.reshape(encoded_img.shape[0], -1)




	kmeans = KMeans(n_clusters=2,init = 'k-means++', random_state=0).fit(encoded_img)

	# In[8]:


	f = pd.read_csv(argv[1])
	IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])


	# In[9]:


	o = open(argv[2],'w')
	o.write("ID,Ans\n")
	for idx, i1, i2 in zip(IDs, idx1, idx2):
		p1 = kmeans.labels_[i1]
		p2 = kmeans.labels_[i2]
		if p1 == p2:
			pred = 1
		else:
			pred = 0
		o.write("{},{}\n".format(idx, pred))
	o.close()


if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))