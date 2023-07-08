from RTVFI import *
import matplotlib.pyplot as plt

rtvfi = RTVFI()
# Show params
batch = 5
params_info = paddle.summary(rtvfi,((1, 6, 360, 640),(1, 6, 360, 640)))
print(params_info)
custom_dataset = Dataset('Super-Resolution/data/1080cut')
print('-'*75,'\ncustom_dataset images: ',len(custom_dataset))

# 用 DataLoader 实现数据加载
train_loader = paddle.io.DataLoader(custom_dataset, batch_size=batch, shuffle=True)
# print("[Dataloader] Done!")

# 将rtsr3模型及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。
rtvfi.train()
# print("[Training] Setting done!")

# 设置迭代次数
epochs = 5

# 设置优化器
optim = paddle.optimizer.Adam(parameters=rtvfi.parameters())
# 设置损失函数
loss_fn = paddle.nn.MSELoss()
acc_fn = Precision()
# set hisfea0, hisfea1
hisfea0 = None
hisfea1 = None
tag0 = 0
tag1 = 0
losslist = []
acclist = []
# acc_fn = paddle.metric.accuracy()
for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        # print("[Training] Date fetching done!")
        x0 = data[0]            # 训练数据
        x1 = data[1]            # 训练数据
        yl = data[2]            # 训练数据标签
        # print(batch_id, x_data.shape, y_data.shape)
        predicts = rtvfi(x0, x1)
        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, yl)
        # # 计算准确率 等价于 prepare 中metrics的设置
        acc_fn.update(predicts, yl)
        acc = acc_fn.accumulate()
        acc_fn.reset()
        # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中
        # 反向传播 
        loss.backward()
        # print("[Training] Backward done!")
        # if (batch_id+1) % 900 == 0:
        print("[Train] epoch: {}, batch_id: {}, loss is: {}, acc is: {:.4f}%".format(epoch, batch_id+1, loss.numpy(), 100*acc))
        losslist.append(loss.numpy())
        acclist.append(acc)
        # 更新参数 
        optim.step()
        # 梯度清零
        optim.clear_grad()

# save
paddle.jit.save(
    layer=rtvfi,
    path="VFI/RTVFI/jit.save/rtvfi",
    input_spec=[
        paddle.static.InputSpec(shape=[1, 6, 360, 640], dtype='float32'), 
        paddle.static.InputSpec(shape=[1, 6, 360, 640], dtype='float32')
        ])

paddle.onnx.export(
    rtvfi, 
    "VFI/RTVFI/onnx.save/rtvfi", 
    input_spec = [
    paddle.static.InputSpec(shape=[1, 6, 360, 640], dtype='float32'), 
    paddle.static.InputSpec(shape=[1, 6, 360, 640], dtype='float32')
    ], 
    opset_version=12)
print("[Model Save] Done!")

# Visualize the loss and acc of training
l = np.array(losslist)
x_axis = len(l)
step = np.linspace(1,x_axis,x_axis)
plt.plot(step,l,label="Train Loss")
plt.legend(loc='upper right')
plt.title('step-loss')
plt.xlim((0, x_axis))
plt.gca().set_ylim(bottom=0)
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('VFI/RTVFI/doc/loss.png', bbox_inches='tight')
plt.show()

a = np.array(acclist)
plt.plot(step,a,label="Train Acc")
plt.legend(loc='upper right')
plt.title('step-acc')
plt.xlim((0, x_axis))
plt.gca().set_ylim(bottom=0)
plt.xlabel('step')
plt.ylabel('acc')
plt.savefig('VFI/RTVFI/doc/acc.png', bbox_inches='tight')
plt.show()

print("[Visualization] Done!")