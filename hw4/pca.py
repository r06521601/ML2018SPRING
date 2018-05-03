
# coding: utf-8

# In[1]:


from skimage import io
import numpy as np
import os

import sys

# In[2]:


# In[3]:
def main(argv):

	X = []
	X_pick = []


	# In[4]:


	for i in range(415):
		img = io.imread(argv[0]+'/'+str(i)+'.jpg')
		X.append(img.flatten())

	img_pick = io.imread(argv[0]+'/'+argv[1])
	X_pick.append(img_pick.flatten())
	# In[5]:


	X = np.array(X)
	X_pick = np.array(X_pick)
	
	


	# In[6]:


	pics_matrix = X.copy()


	# In[7]:



	mu = np.mean(pics_matrix, axis=0)
	x = pics_matrix - mu
	X_pick = X_pick - mu
	eigen_faces, sigma, v = np.linalg.svd(x.T, full_matrices=False)

	# In[13]:

	picked_faces = eigen_faces[:,:4]


	# In[19]:



	pic = np.dot(X_pick, picked_faces)
	pics = np.dot(pic, picked_faces.T)
	pics += mu.flatten()
	pics -= np.min(pics)
	pics /= np.max(pics)

	pics = (pics * 255).astype(np.uint8)

	io.imsave('reconstrcution.jpg' ,pics.reshape(600,600,3))






# In[133]:

if __name__ == '__main__':  
    sys.exit(main(sys.argv[1:]))


