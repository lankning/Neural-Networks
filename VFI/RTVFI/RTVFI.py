from tkinter import Image
import paddle
import numpy as np
import os, cv2
from math import pi
# Subclass mode: https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_cn.html
# API Overview： https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html

class ConVFI(paddle.nn.Layer):
    def __init__(self, 
                kernels = 32,
                ):
        '''
        Function: Initialize all variables.
        '''
        super(ConVFI, self).__init__()
        encoderlist = []
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = 12, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=3, kernel_size=3, padding="SAME"
                ))
        self.encoderlist =  paddle.nn.LayerList(encoderlist)

        self.silu = paddle.nn.Silu("Silu")

    def forward(self, F0, F1):
        '''
        Function: Forward calculate the features of network.
        Structure: 
        1. Encoder: convolution layers, extract the features of inputs
        '''
        # encoder part
        x = paddle.concat([F0, F1], 1)
        x = self.silu(self.encoderlist[0](x))
        x = self.silu(self.encoderlist[1](x))
        x = self.silu(self.encoderlist[2](x))
        x = self.silu(self.encoderlist[3](x))
        x = self.silu(self.encoderlist[4](x))
        x = self.silu(self.encoderlist[5](x))
        return x

class RTVFI(paddle.nn.Layer):
    def __init__(self, 
                kernels = 32, 
                n = 1,
                h = 360, 
                w = 640
                ):
        '''
        Function: Initialize all variables.
        '''
        super(RTVFI, self).__init__()
        encoderlist = []
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = 6, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        encoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        self.encoderlist =  paddle.nn.LayerList(encoderlist)

        decoderlist = []
        decoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        decoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        decoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        decoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"
                ))
        decoderlist.append(
            paddle.nn.Conv2D(
                in_channels = kernels, out_channels=3, kernel_size=3, padding="SAME"
                ))
        self.decoderlist =  paddle.nn.LayerList(decoderlist)

        self.silu = paddle.nn.Silu("Silu")

        self.n = n
        self.h = h
        self.w = w

        # self.codes = paddle.zeros((self.n, 1, self.h, self.w))
        # self.codes += 0.5
        self.codes = self.get_temporary_codes()

    
    def get_temporary_codes(self, timeid = 0.5):
        '''
        Function: Get the internel temporary codes of 
                  the frame to be rebuilt.
        '''
        codes = paddle.zeros((self.n, 1, self.h, self.w))
        codes[:,:,:,:] = timeid
        return codes


    def forward(self, F0, F1):
        '''
        Function: Forward calculate the features of network.
        Structure: 
        0. Temporary and spatio ecoding
        1. Encoder: convolution layers, extract the features of inputs
        2. Decoder: convolution layers, rebuild the frame
        '''
        # encoder part
        F0 = self.silu(self.encoderlist[0](F0))
        F1 = self.silu(self.encoderlist[0](F1))
        F0 = self.silu(self.encoderlist[1](F0))
        F1 = self.silu(self.encoderlist[1](F1))
        F0 = self.silu(self.encoderlist[2](F0))
        F1 = self.silu(self.encoderlist[2](F1))
        F0 = self.silu(self.encoderlist[3](F0))
        F1 = self.silu(self.encoderlist[3](F1))
        F0 = self.silu(self.encoderlist[4](F0))
        F1 = self.silu(self.encoderlist[4](F1))

        # extract features needed
        F0 = paddle.multiply(F0, self.codes)
        F1 = paddle.multiply(F1, self.codes)
        res = F0 + F1

        # decoder part: rebuild the frame
        res = self.silu(self.decoderlist[0](res))
        res = self.silu(self.decoderlist[1](res))
        res = self.silu(self.decoderlist[2](res))
        res = self.silu(self.decoderlist[3](res))
        res = self.silu(self.decoderlist[4](res))
        return res

class Dataset(paddle.io.Dataset):
    '''
    Step 1: inherit the class paddle.io.Dataset
    '''
    def __init__(self, 
                indir,
                h = 360, 
                w = 640
                ):
        '''
        Step 2: realize __init__()
        initialize the dataset,
        mapping the samples and labels to list
        '''
        super(Dataset, self).__init__()
        self.data_list = []
        filelist = os.listdir(indir)
        filelist.sort(key=lambda x:int(x[5:-4]))
        for i, filename in enumerate(filelist):
            if (i != 0) and (i < len(filelist)-1):
                self.data_list.append([
                    os.path.join(indir, filelist[i-1]), 
                    os.path.join(indir, filename),
                    os.path.join(indir, filelist[i+1])
                    ])
        codes = np.zeros((3, h, w))
        # Temporaray codes
        codes[0, :, :] = 0
        # Line direction
        for j in range(w):
            codes[1, :, j] = np.sin(j/w)
        # Row direction
        for j in range(h):
            codes[2, j, :] = np.sin(j/h)
        self.encodes = {"codes": codes, "length": 2}

    def temporary_and_spatial_encoding(self, *args):
        '''
        Function: This function is designed to encode
                  the tensor by axises of time, line 
                  and row.
        ---------------------------------------------
        Activate Function: sin

        Input:  numpy.array, 3-dims, data 
                format = CHW, shape = [C,H,W]
        Output: numpy.array, 3-dims, data 
                format = CHW, shape = [C+3, H, W]
        ---------------------------------------------
        Inner parameters:
        1. length = length of list
        1. h: height
        2. w: width
        '''
        if args[0].shape==self.encodes['codes'].shape:
            _, self.h, self.w = self.encodes['codes'].shape
        else:
            _, self.h, self.w = args[0].shape
            codes = np.zeros((3, self.h, self.w))
            # Temporaray codes
            codes[0, :, :] = 0
            # Line direction
            for j in range(self.w):
                codes[1, :, j] = np.sin(j/self.w)
            # Row direction
            for j in range(self.h):
                codes[2, j, :] = np.sin(j/self.h)
            self.encodes['codes'] = codes
        
        self.encodes['length'] = len(args)

        rtn = []
        for i, image in enumerate(args):
            self.encodes['codes'][0, :, :] = np.sin(i/(self.encodes['length']-1))
            image = np.concatenate([image, self.encodes['codes']], 0)
            rtn.append(image)
        return rtn

    def __getitem__(self, index):
        '''
        Step 3: realize __get_item__()
        Input: index
        RTN: data[index], label[index]
        '''
        # 根据索引，从列表中取出一个图像
        # print("[Dataset::__getitem__]",self.data_list[index])
        in0path, outpath, in1path = self.data_list[index]
        # 读取彩色图
        in0 = cv2.imread(in0path,  cv2.IMREAD_COLOR)
        in1 = cv2.imread(in1path,  cv2.IMREAD_COLOR)
        out = cv2.imread(outpath,  cv2.IMREAD_COLOR)
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        in0 = in0.astype('float32').transpose((2,0,1))
        in1 = in1.astype('float32').transpose((2,0,1))
        out = out.astype('float32').transpose((2,0,1))
        in0, in1 = self.temporary_and_spatial_encoding(in0, in1)
        in0 = in0.astype('float32')
        in1 = in1.astype('float32')
        # 返回图像和对应标签
        return in0, in1, out

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


class Precision(paddle.metric.Metric):
    """
    Precision (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances. Refer to
    https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers

    Noted that this class manages the precision score only for binary
    classification task.
    
    ......

    """

    def __init__(self, name='precision', *args, **kwargs):
        super(Precision, self).__init__(*args, **kwargs)
        self.tp = 0  # true positive
        self.sum = 0  # sum elements
        self._name = name

    def update(self, preds, labels):
        """
        Update the states based on the current mini-batch prediction results.

        Args:
            preds (numpy.ndarray): The prediction result, usually the output
                of two-class sigmoid function. It should be a vector (column
                vector or row vector) with data type: 'float64' or 'float32'.
            labels (numpy.ndarray): The ground truth (labels),
                the shape should keep the same as preds.
                The data type is 'int32' or 'int64'.
        """
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        elif isinstance(preds, np.array):
            None
        else:
            raise ValueError("The 'preds' must be a numpy ndarray or Tensor.")

        if isinstance(labels, paddle.Tensor):
            labels = labels.numpy()
        elif isinstance(labels, np.array):
            None
        else:
            raise ValueError("The 'labels' must be a numpy ndarray or Tensor.")

        self.sum = np.prod(np.shape(labels))
        preds = np.round(preds).astype("int32")
        labels = np.round(labels).astype("int32")
        self.tp = np.sum([preds==labels])

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.tp = 0
        self.sum = 0

    def accumulate(self):
        """
        Calculate the final precision.

        Returns:
            A scaler float: results of the calculated precision.
        """
        return float(self.tp/self.sum)

    def name(self):
        """
        Returns metric name
        """
        return self._name

if __name__=="__main__":
    rtvfi = RTVFI()
    # rtvfi = ConVFI()
    # Show params
    params_info = paddle.summary(rtvfi,((1, 6, 360, 640),(1, 6, 360, 640)))
    print(params_info)
    custom_dataset = Dataset('Super-Resolution/data/1080cut')
    print('-'*75,'\ncustom_dataset images: ',len(custom_dataset))