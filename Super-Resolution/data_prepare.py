#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils import *
import os


# In[5]:


# check the existence of folder
dir_list = ['./data','./data/540','./data/1080']
for folder in dir_list:
    if bool(1-os.path.exists(folder)):
        os.mkdir(folder)
        print("No target folder %s, create it."%folder)


# In[6]:


# get frames from video with interval of 1 frames
getFrame("./data/1080.mp4", "./data/1080/", 1)
print("getFrame finished.")


# In[7]:


# check the existence of folder
dir_list = ['./data','./data/540cut','./data/1080cut']
for folder in dir_list:
    if bool(1-os.path.exists(folder)):
        os.mkdir(folder)
        print("No target folder %s, create it."%folder)


# In[9]:


CutList("./data/1080/","./data/1080cut/",1/3,2/3,1/3,2/3)
print("get 1/3 cut pics finished.")


# In[10]:


# get LR pics from HR pics by 1/2
blurryList("./data/1080cut/","./data/540cut/",2)
print("get 1/3 LR pics finished.")
# # get LR pics from HR pics by 1/4
# blurryList("./data/cut_data/","./data/cut_data4x/",4)
# print("get 1/4 LR pics finished.")


# In[ ]:


# # get LR pics from HR pics by 1/2
# blurryList("./data/train_data/","./data/train_data2x/",2)
# print("get 1/2 LR pics finished.")
# # get LR pics from HR pics by 1/4
# blurryList("./data/train_data/","./data/train_data4x/",4)
# print("get 1/4 LR pics finished.")


# In[ ]:




