import paddlehub as hub
import cv2, time, os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


stylepro_artistic = hub.Module(name="stylepro_artistic")
styleimg = cv2.imread('style4.jpg')

capture=cv2.VideoCapture(0)
fps = 0
t1 = time.time()
while(True):
    # t1 = time.time()
    # 读取某一帧
    fps = fps + 1
    ref,frame=capture.read()

    result = stylepro_artistic.style_transfer(images=[{
                                                'content': frame,
                                                'styles': [styleimg]
                                                     }],
                                            use_gpu = True)

    frame = result[0]['data'].astype('uint8')

    # fps.append(( fps + (1./(time.time()-t1)) ) / 2)
    # time_his.append(time.time() - t1)
    # del time_his[0]
    frame = cv2.putText(frame, "fps= %.2f"%(1/((time.time() - t1)/fps)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)
    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        break