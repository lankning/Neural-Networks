import paddlehub as hub
import cv2, time, os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型加载
# model = hub.Module(name='MiDaS_Small', use_gpu=True)
model = hub.Module(name='MiDaS_Large', use_gpu=True)

capture=cv2.VideoCapture(0)
fps = 0
t1 = time.time()
while(True):
    # t1 = time.time()
    # 读取某一帧
    fps = fps + 1
    ref,frame=capture.read()

    result = model.depth_estimation(images=[frame])

    frame = result[0].astype('uint8')

    # fps.append(( fps + (1./(time.time()-t1)) ) / 2)
    # time_his.append(time.time() - t1)
    # del time_his[0]
    frame = cv2.putText(frame, "fps= %.2f"%(1/((time.time() - t1)/fps)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)
    c = cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break