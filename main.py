#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Tuple

import numpy as np
from itertools import product
import cv2


# In[ ]:





# In[ ]:





# In[2]:


#img = cv2.imread('P:\\DataScience\\motion_blur\\data\\5.jpg')
#img = cv2.imread('P:\\DataScience\\motion_blur\\data\\12.webp')
img = cv2.imread('P:\\DataScience\\motion_blur\\data\\4.jpg', cv2.IMREAD_UNCHANGED)


# In[3]:


cv2.imshow('current image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[4]:


img.shape


# In[ ]:





# In[5]:


type(img)


# In[6]:


img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[7]:


cv2.imshow('current image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[8]:


def add_color(src: np.ndarray, dest: np.ndarray, 
             pt_lu: Tuple[int, int], pt_rb: Tuple[int, int]
            ) -> np.ndarray:
    """
    """
    add_img = src[pt_lu[1]: pt_rb[1], pt_lu[0]:pt_rb[0], :]
    dest[pt_lu[1]: pt_rb[1], pt_lu[0]:pt_rb[0], :] = add_img
    
    return dest



def add_gray(src: np.ndarray, dest: np.ndarray, 
             pt_lu: Tuple[int, int], pt_rb: Tuple[int, int]
            ) -> np.ndarray:
    """
    """
    add_img = src[pt_lu[1]: pt_rb[1], pt_lu[0]:pt_rb[0], :]
    add_img = cv2.cvtColor(add_img, cv2.COLOR_BGR2GRAY)
    dest[pt_lu[1]: pt_rb[1], pt_lu[0]:pt_rb[0], 0] = add_img
    dest[pt_lu[1]: pt_rb[1], pt_lu[0]:pt_rb[0], 1] = add_img
    dest[pt_lu[1]: pt_rb[1], pt_lu[0]:pt_rb[0], 2] = add_img

    return dest


# In[ ]:





# In[ ]:





# In[9]:


pt1 = (420, 5)
pt2 = (600, 405)


# In[10]:


pt1[0], pt2[0], pt1[1], pt2[1]


# In[11]:


img.shape[0]


# In[12]:


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_runners = img[pt1[1]: pt2[1], pt1[0]:pt2[0], :].copy()
#img3 = cv2.rectangle(img3, pt1, pt2, color=(0, 255, 0), thickness=2)

img_mesh = np.zeros_like(img)

img_mesh[:,:,0] = img_gray
img_mesh[:,:,1] = img_gray
img_mesh[:,:,2] = img_gray

#img_mesh[pt1[1]: pt2[1], pt1[0]:pt2[0], :] = img_runners
img_mesh = add_color(img, img_mesh, pt1, pt2)
#img_mesh = add_gray(img, img_mesh, pt1, pt2)


# In[13]:


#cv2.imshow('current image', img_gray)
cv2.imshow('current image', img_mesh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:





# In[ ]:




