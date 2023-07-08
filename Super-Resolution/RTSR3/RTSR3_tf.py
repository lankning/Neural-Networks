import tensorflow as tf
import numpy as np
from PIL import Image
import os

def loadimg(imgpath):
    img = np.array(Image.open(imgpath))
    return img

def generate_data(inpath, outpath):
    # read and sort train data from dir
    lr_list = os.listdir(inpath)
    lr_list.sort(key=lambda x:int(x[5:-4]))#倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    lr_list = [os.path.join(inpath,c) for c in lr_list]
    gt_list = os.listdir(outpath)
    gt_list.sort(key=lambda x:int(x[5:-4]))
    gt_list = [os.path.join(outpath,c) for c in gt_list]
    i = 0
    while True:
        print("\nTraining data index: %d" % i, lr_list[i], gt_list[i])
        lr = np.array([loadimg(lr_list[i])], dtype=np.float32)
        gt = np.array([loadimg(gt_list[i])], dtype=np.float32)
        i += 1
        yield lr, gt

tf.compat.v1.disable_eager_execution()

class RTSR3(tf.keras.models.Model):
    def __init__(self, scale = 2, L=3):
        super(RTSR3, self).__init__()
        self.L = L
        self.scale = scale
        self.convlist = []
        for i in range(L):
            self.convlist.append(tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='tanh'))
        self.fea_his = [tf.constant(np.zeros((1,180,390,32),dtype=np.float32)),tf.constant(np.zeros((1,180,390,32),dtype=np.float32))]
        tf.stop_gradient(self.fea_his[0])
        tf.stop_gradient(self.fea_his[1])

        self.linear_0 = tf.keras.layers.Conv2D(3*scale*scale, 3, strides=1, padding='same', activation=None)
        self.linear_1 = tf.keras.layers.Conv2D(3*scale*scale, 3, strides=1, padding='same', activation=None)

        self.nonlinear_0 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='tanh')
        self.nonlinear_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='tanh')

        self.his1_nonlinear_0 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='tanh')
        self.his1_nonlinear_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='tanh')

        self.his2_nonlinear_0 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='tanh')
        self.his2_nonlinear_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='tanh')

        self.unify_channel = tf.keras.layers.Conv2D(3*scale*scale, 3, strides=1, padding='same', activation='tanh')

        self.fea_fix = tf.keras.layers.Conv2D(3, 3, strides=1, padding='same', activation='tanh')

    def update_his_fea(self, nonfea):
        if len(self.fea_his)==2:
            self.fea_his.pop()
        self.fea_his.insert(0,nonfea)
        tf.stop_gradient(self.fea_his[0])
        tf.stop_gradient(self.fea_his[1])

    def clear_fea(self):
        self.fea_his = []
 
    def call(self, x):
        for i in range(self.L):
            x = self.convlist[i](x)
        # Linear feature
        linear_fea = self.linear_1(self.linear_0(x))
        # Nonlinear feature
        nonlinear_fea = self.nonlinear_1(self.nonlinear_0(x))
        his_fea1 = self.his1_nonlinear_1(self.his1_nonlinear_0(self.fea_his[0]))
        his_fea2 = self.his2_nonlinear_1(self.his2_nonlinear_0(self.fea_his[1]))
        
        non_fea = tf.concat([nonlinear_fea,his_fea1,his_fea2],axis=-1)
        # non_fea = nonlinear_fea
        self.update_his_fea(nonlinear_fea)
        non_fea = self.unify_channel(non_fea)

        res = tf.nn.depth_to_space(non_fea+linear_fea, block_size=self.scale, data_format='NHWC', name=None)
        res = self.fea_fix(res)
        return res

if __name__=="__main__":
    rtsr3 = RTSR3(2,3)
    rtsr3.build(input_shape=(1,100,100,3))
    print(rtsr3.summary())