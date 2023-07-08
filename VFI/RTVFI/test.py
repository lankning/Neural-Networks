from RTVFI import *

rtvfi = paddle.jit.load("VFI/RTVFI/jit.save/rtvfi")
rtvfi.eval()
params_info = paddle.summary(rtvfi,((1, 6, 360, 640),(1, 6, 360, 640)))
print(params_info)
custom_dataset = Dataset('Super-Resolution/data/1080cut')
print('-'*75,'\ncustom_dataset images: ',len(custom_dataset))

train_loader = paddle.io.DataLoader(custom_dataset, batch_size=1, shuffle=False)

out = cv2.VideoWriter('VFI/RTVFI/doc/out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 360*2))

for id, data in enumerate(train_loader()):
    x0 = data[0]
    x1 = data[1]
    yl = data[2]
    predicts = rtvfi(x0, x1)
    predicts = predicts.numpy()[0].transpose((1,2,0)).astype('uint8')
    x0 = x0.numpy()[0,:3,:,:].transpose((1,2,0)).astype('uint8')
    x1 = x1.numpy()[0,:3,:,:].transpose((1,2,0)).astype('uint8')
    yl = yl.numpy()[0,:3,:,:].transpose((1,2,0)).astype('uint8')
    if id % 2==0:
        concat = np.concatenate([x1, predicts], 0)
        cv2.imshow("video", concat)
        out.write(concat)
        c = cv2.waitKey(1) & 0xff 
        if c==27:
            out.release()
            break
    else:
        concat = np.concatenate([yl, predicts], 0)
        cv2.imshow("video", concat)
        out.write(concat)
        c = cv2.waitKey(1) & 0xff 
        if c==27:
            out.release()
            break