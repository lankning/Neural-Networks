import paddle, cv2, time
import numpy as np

rtsr3 = paddle.jit.load("jit.save/rtsr3")
rtsr3.eval()
params_info = paddle.summary(rtsr3,(1, 3, 180, 320))
print(params_info)

def psnr(x,y):
    mse = np.mean(np.square(x-y))
    psnr = 10*np.log10(255*255/mse)
    return psnr

capture=cv2.VideoCapture('../data/1080.mp4')
fs = 0

while(True):
    fs += 1
    ref,frame=capture.read()
    if ref:
        # cut
        frame = frame[360:720,640:1280,:]
        # copy to calculate psnr
        infra = frame
        if fs == 1:
            # out >> .mp4
            out = cv2.VideoWriter('doc/out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1],2*frame.shape[0]))
            # t0 = time.time()

        # RTSR3: resize to LR (input)
        frame = cv2.resize(frame, (320,180))
        frame = frame.astype('float32').transpose((2,0,1))
        frame = np.array([frame])
        pred = rtsr3(frame)[0][0]
        pred = pred.numpy()
        pred = pred.transpose((1,2,0))
        pred = np.uint8(pred)
        pred = pred.copy()
        # pred = cv2.putText(pred, "PSNR=%.2f"%(psnr(pred,infra)), (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        pred = cv2.putText(pred, "RTSR3", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if fs == 200:
            cv2.imwrite("doc/out_rtsr3.png", pred)
        # t1 = time.time()
        # pred = cv2.putText(pred, "fps= %.2f"%(1/((t1 - t0)/fs)), (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # resize input to bilinear
        frame = frame[0].transpose((1,2,0))
        frame = cv2.resize(frame,(640,360),interpolation=cv2.INTER_LINEAR)
        frame = np.uint8(frame)
        frame = frame.copy()
        # frame = cv2.putText(frame, "PSNR=%.2f"%(psnr(frame,infra)), (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.putText(frame, "Bilinear", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if fs == 200:
            cv2.imwrite("doc/out_bilinear.png", frame)
        # t2 = time.time()
        # frame = cv2.putText(frame, "fps= %.2f"%(1/(t2-t1)), (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # concat the bilinear and HR together
        outframe = np.concatenate((frame,pred), axis=0)
        outframe = outframe.copy()
        cv2.imshow("video",outframe)
        out.write(outframe)
        c= cv2.waitKey(1) & 0xff 
        if c==27:
            capture.release()
            out.release()
            break
    else:
        break