#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2, math, os
from PIL import Image
import numpy as np
import tensorflow as tf


# In[ ]:


# Function: Get frames from video with interval(gapFrame)
def getFrame(videoPath, savePath, gapFrame=1):
# this code is original from https://blog.csdn.net/u010555688/article/details/79182362
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # frame = np.rot90(frame)
                #cv2.imshow('video', frame)
                numFrame += 1
                #print(numFrame)
                if (numFrame%gapFrame==0):
                    newPath = savePath + 'Frame{:04d}.png'.format(numFrame)
                    cv2.imencode('.png', frame)[1].tofile(newPath)
        else:
            break


# In[ ]:


# get LR pics from HR pics by 1/blurryTimes with mode "select"
def blurryList(imgFolder, svFolder, blurryTimes=4):#, mode="select"
    img_list = os.listdir(imgFolder)
    for imgName in img_list:
        imgPath = os.path.join(imgFolder,imgName)
        # blurry(imgPath,svFolder,blurryTimes,mode)
        img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        w,h,d = img.shape
        img = cv2.resize(img, (h//blurryTimes, w//blurryTimes),interpolation=cv2.INTER_NEAREST)
        svpath = os.path.join(svFolder,imgName)
        cv2.imwrite(svpath, img)

# blurryList("./data/train_data/","./data/train_data4x/",4)#,"select"


# In[ ]:


# Function: Get all images from specified dir, return [None,?,?,3]
def load_images(toreadPath,split=4):
    imgList = []
    file_list = os.listdir(toreadPath)
    file_list.sort(key=lambda x:int(x[5:-split]))#倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    # print(file_list)
    for imgName in file_list:
        imgPath = os.path.join(toreadPath,imgName)
        img = Image.open(imgPath)
        img_arr = np.array(img)
        imgList.append(img_arr)
    return np.array(imgList)


# In[ ]:


# BiCubic Interpolation
def BiCubicList(imgdata,k):
    num,a,b,c = imgdata.shape
    bigImgs = np.zeros((num,k*a,k*b,c))
    for i in range(num):
        # use the implementation above
        # bigImgs.append(BiCubic(imgdata[i,...],k))
        # For efficiency, we use Pillow.Image.BICUBIC() method
        rgb = Image.fromarray(np.uint8(imgdata[i,...]),"RGB")
        rgb = rgb.resize((b*k,a*k), Image.BICUBIC)
        bigImgs[i,...] = np.array(rgb)
    return bigImgs


# In[ ]:


# Function: Get all images from specified dir, return [None,?,?,3]
def load_images_bybicubic(toreadPath,k=2,split=4):
    imgList = []
    file_list = os.listdir(toreadPath)
    file_list.sort(key=lambda x:int(x[:-split]))#倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    # print(file_list)
    for imgName in file_list:
        imgPath = os.path.join(toreadPath,imgName)
        img = Image.open(imgPath)
        img_arr = np.array(img)
        a,b,c = img_arr.shape
        rgb = Image.fromarray(np.uint8(img_arr),"RGB")
        rgb = rgb.resize((b*k,a*k), Image.BICUBIC)
        img_arr = np.array(rgb)
        imgList.append(img_arr)
    return np.float32(np.array(imgList))


# In[ ]:


def loadimgs_from_paths(x_paths,y_paths):
    xList=[]
    for x_path in x_paths:
        img = Image.open(x_path)
        img_arr = np.array(img)
        xList.append(img_arr)
    yList=[]
    for y_path in y_paths:
        img = Image.open(y_path)
        img_arr = np.array(img)
        yList.append(img_arr)
    return np.array(xList),np.array(yList)


# In[ ]:


def images_to_video(imgpath,svpath,videoname="demo",fps=60):
    img_array = []
    images_list = os.listdir(imgpath)
    images_list.sort(key=lambda x:int(x[5:-4]))
    for i in images_list:
        filename = os.path.join(imgpath,i)
        print(filename)
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(svpath,'%s.avi'%videoname), cv2.VideoWriter_fourcc('X','V','I','D'), fps, (img.shape[1],img.shape[0]))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


# In[ ]:


class MySR(tf.keras.models.Model):
    def __init__(self,upscale):
        # 调用父类__init__()方法
        super(MySR, self).__init__()
        self.linear_conv = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation=None, name="linear_conv")
        self.conv2d_1 = tf.keras.layers.Conv2D(3*upscale*upscale, 5, strides=1, padding='same', activation='tanh', name="feature_extraction_1")
        self.conv2d_2 = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation='tanh', name="feature_extraction_2")
        #self.conv2d_3 = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation='tanh', name="feature_extraction_3")
        self.conv2d_4 = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation='tanh', name="motion_wrap1")
        self.conv2d_5 = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation='tanh', name="motion_wrap2")
        self.conv2d_6 = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation='tanh', name="motion_wrap3")
        self.conv2d_7 = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation='tanh', name="motion_wrap4")
        self.conv2d_8 = tf.keras.layers.Conv2D(3*upscale*upscale, 3, strides=1, padding='same', activation='tanh', name="conv1")
        self.conv2d_9 = tf.keras.layers.Conv2D(3, 3, strides=1, padding='same', activation=None, name="conv2")
        self.r = upscale
        self.shuffle = Shuffle(upscale)
        #print("Model inited.")

    def call(self, inputs):
        inputs = inputs/127.5 - 1
        x1,x2,x3 = tf.split(inputs,3,1)
        x1 = tf.squeeze(x1, axis=1)
        x2 = tf.squeeze(x2, axis=1)
        x3 = tf.squeeze(x3, axis=1) # current frame
        # Linear Feature Extraction
        linear_fea = self.linear_conv(x3)
        # Nonlinear Feature Extraction
        x1 = self.conv2d_1(x1)
        x1 = self.conv2d_2(x1)
        #x1 = self.conv2d_3(x1)
        x2 = self.conv2d_1(x2)
        x2 = self.conv2d_2(x2)
        #x2 = self.conv2d_3(x2)
        x3 = self.conv2d_1(x3)
        x3 = self.conv2d_2(x3)
        #x3 = self.conv2d_3(x3)
        # Motion Wrap
        x2 = self.conv2d_4(x2)
        x2 = self.conv2d_5(x2)
        x1 = self.conv2d_6(x1)
        x1 = self.conv2d_7(x1)
        # Feature Concat
        nonlinear_fea = tf.concat([x1,x2,x3],axis=3)
        nonlinear_fea = self.conv2d_8(nonlinear_fea)
        x = nonlinear_fea + linear_fea
        # Pixel Shuffle
        x = self.shuffle(x)
        x = self.conv2d_9(x)
        x = tf.clip_by_value(x, -1, 1)
        x = 127.5*(x+1)
        return x
# In[ ]:


def con_img2v(path1,path2,svpath,axis=1,videoname="demo",fps=60):
    img_array = []
    img_list1 = os.listdir(path1)
    img_list1.sort(key=lambda x:int(x[5:-4]))
    img_list1 = img_list1[:6000]
    img_list2 = os.listdir(path2)
    img_list2.sort(key=lambda x:int(x[5:-4]))
    img_list2 = img_list2[:6000]
    num = min(len(img_list1),len(img_list2))
    for i in range(num):
        if i % 20 == 0:
            print("processing: %.2f%%" % (100*i/num))
        file1 = os.path.join(path1,img_list1[i])
        img1 = cv2.imread(file1)
        file2 = os.path.join(path2,img_list2[i])
        img2 = cv2.imread(file2)
        if (img1 is None) or (img2 is None):
            print(file1, file2 + " is error!")
            continue
        img = np.append(img1,img2,axis=axis)
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(svpath,'%s.avi'%videoname), cv2.VideoWriter_fourcc('X','V','I','D'), fps, (img.shape[1],img.shape[0]))
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
# con_img2v("temp_INTER_LINEAR","temp_sr","./",1,"concat",20)


# In[ ]:


class Shuffle(tf.keras.layers.Layer):
    def __init__(self,r):
        # 调用父类__init__()方法
        super(Shuffle, self).__init__()
        self.r = r

    def call(self, inputs):
        x_c = []
        for c in range(3):
            t = inputs[:,:,:,c*self.r*self.r:c*self.r*self.r+self.r*self.r] # [B,H,W,R*R]
            t = tf.compat.v1.depth_to_space(t, self.r) # [B,H*R,W*R,1]
            x_c += [t]
        x = tf.concat(x_c, axis=3)   # [B,H*R,W*R,3]
        return x





