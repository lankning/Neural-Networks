import paddle
import numpy as np
import os, cv2

# Subclass mode: https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_cn.html
# API Overview： https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Overview_cn.html

class RTSR3(paddle.nn.Layer):
    def __init__(self, scale = 2, kernels=32):
        super(RTSR3, self).__init__()
        self.scale = scale
        convlist = []
        convlist.append(paddle.nn.Conv2D(in_channels = 3,  out_channels=kernels, kernel_size=3, padding="SAME"))
        convlist.append(paddle.nn.Conv2D(in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"))
        convlist.append(paddle.nn.Conv2D(in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME"))
        self.convlist =  paddle.nn.LayerList(convlist)

        self.linear0 = paddle.nn.Conv2D(in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME")
        self.linear1 = paddle.nn.Conv2D(in_channels = kernels, out_channels=3*scale*scale, kernel_size=3, padding="SAME")

        self.nonlinear0 = paddle.nn.Conv2D(in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME")
        self.nonlinear1 = paddle.nn.Conv2D(in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME")
        self.n_tanh0 = paddle.nn.Tanh()
        self.n_tanh1 = paddle.nn.Tanh()

        self.wrap0_0 = paddle.nn.Conv2D(in_channels = kernels * 2, out_channels=kernels, kernel_size=3, padding="SAME")
        self.wrap0_1 = paddle.nn.Conv2D(in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME")
        self.w0_tanh0 = paddle.nn.Tanh()
        self.w0_tanh1 = paddle.nn.Tanh()

        self.wrap1_0 = paddle.nn.Conv2D(in_channels = kernels * 2, out_channels=kernels, kernel_size=3, padding="SAME")
        self.wrap1_1 = paddle.nn.Conv2D(in_channels = kernels, out_channels=kernels, kernel_size=3, padding="SAME")
        self.w1_tanh0 = paddle.nn.Tanh()
        self.w1_tanh1 = paddle.nn.Tanh()

        self.unify_channel = paddle.nn.Conv2D(in_channels = 3 * kernels, out_channels=3*scale*scale, kernel_size=3, padding="SAME")
        self.u_tanh = paddle.nn.Tanh()

        self.shuffle = paddle.nn.PixelShuffle(scale)

        self.fea_fix = paddle.nn.Conv2D(in_channels = 3, out_channels=3, kernel_size=3, padding="SAME")
        self.f_tanh = paddle.nn.Tanh()

        self.upsample = paddle.nn.Upsample(scale_factor=scale, mode="bilinear")

    def init_his_fea(self, fea):
        return paddle.zeros_like(fea)   

    # 执行前向计算
    def forward(self, x, his_fea0=None, his_fea1=None, tag0=0, tag1=0):
        x_up = self.upsample(x)
        x = x/127.5 - 1
        for i in range(len(self.convlist)):
            if i == 0:
                x = self.convlist[i](x)
            else:
                x = self.convlist[i](x) + x
        linear_fea = self.linear1(self.linear0(x))
        nonlin_fea = self.n_tanh1(self.nonlinear1(self.n_tanh0(self.nonlinear0(x))))
        if tag0 == 0:
            his_fea0 = self.init_his_fea(nonlin_fea)
            his_fea1 = self.init_his_fea(nonlin_fea)
            tag0 = 1
        elif tag1 == 0:
            his_fea1 = self.init_his_fea(nonlin_fea)
            tag1 = 1
        
        # Motion wrap
        rtn1 = his_fea0
        his_fea0 = self.w0_tanh1(self.wrap0_1(self.w0_tanh0(self.wrap0_0(paddle.concat([nonlin_fea,his_fea0], 1)))))
        his_fea1 = self.w1_tanh1(self.wrap1_1(self.w1_tanh0(self.wrap1_0(paddle.concat([nonlin_fea,his_fea1], 1)))))

        # Feature fusion: 0
        non_fea = paddle.concat([nonlin_fea,his_fea0,his_fea1],1)
        non_fea = self.u_tanh(self.unify_channel(non_fea)) # in channels: 32*3

        # # Feature fusion: 1
        # non_fea = nonlin_fea + his_fea0 + his_fea1
        # non_fea = self.u_tanh(self.unify_channel(non_fea)) # in channels: 32

        res = self.shuffle(linear_fea+non_fea)
        res = self.f_tanh(self.fea_fix(res))
        # res = self.fea_fix(res)

        res = 127.5 * (res + 1)

        res = (res + x_up)/2

        return res, nonlin_fea, rtn1, tag0, tag1

class SrDataset(paddle.io.Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, indir, outdir):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(SrDataset, self).__init__()
        self.data_list = []
        filelist = os.listdir(indir)
        for filename in filelist:
            if ".png" in filename:
                self.data_list.append([os.path.join(indir,filename),os.path.join(outdir,filename)])
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        # print("[Dataset::__getitem__]",self.data_list[index])
        inpath, outpath = self.data_list[index]
        # 读取彩色图
        in0 = cv2.imread(inpath,  cv2.IMREAD_COLOR)
        out = cv2.imread(outpath, cv2.IMREAD_COLOR)
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        in0 = in0.astype('float32').transpose((2,0,1))
        out = out.astype('float32').transpose((2,0,1))
        # 返回图像和对应标签
        return in0, out

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
    rtsr3 = RTSR3(2)
    # Show params
    params_info = paddle.summary(rtsr3,(1, 3, 180, 390))
    print(params_info)
    custom_dataset = SrDataset('../data/540cut','../data/1080cut')
    print('-'*75,'\ncustom_dataset images: ',len(custom_dataset))