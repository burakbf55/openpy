import cv2
import numpy as np

cap = cv2.VideoCapture("bg_sample.mp4")

ret, frame = cap.read()


myVideo = cv2.VideoWriter("subtractor.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 30, (int(frame.shape[1]), int(frame.shape[0])))

bySubtraction = cv2.BackgroundSubtractorMOG2(detectShadows = False)

while True:
    ret, frame = cap.read()
    
    if ret:
        mask = bySubtraction.apply(frame, learningRate = -1)

        cv2.imshow('mask', mask)

        fg = cv2.copyTo(frame, mask)
        kernel = np.ones((5,5), np.unint8)
        
        dilation = cv2.dilate(mask, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)

        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

        fg = cv2.copyTo(frame, opening)
        cv2.imshow('BGSUB', fg)
        
        myVideo.write(fg)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xff == ord(q):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()