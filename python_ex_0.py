import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('trackbars')

cv2.createTrackbar('H_min','Trackbars',0,179,nothing)
cv2.createTrackbar('H_max','Trackbars',179,179,nothing)
cv2.createTrackbar('S_min','Trackbars',0,255,nothing)
cv2.createTrackbar('S_max','Trackbars',255,255,nothing)
cv2.createTrackbar('V_min','Trackbars',0,255,nothing)
cv2.createTrackbar('V_max','Trackbars',255,255,nothing)

cap = cv2.VideoCapture(0)

if not cap is open():
    print('열 수 없습니다.')
    exit()

while True:
    ret,frame = cap.read()

    if not ret:
        print('카메라 프레임을 가져올 수 없습니다')
        break
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGRHSV)

    h_min = cv2.getTrackbarPos('H_min','Trackbars')
    h_max = cv2.getTrackbarPos('H_max','Trackbars')
    s_min = cv2.getTrackbarPos('S_min','Trackbars')
    s_max = cv2.getTrackbarPos('s_max','Trackbars')
    v_min = cv2.getTrackbarPos('V_min','Trackbars')
    v_max = cv2.getTrackbarPos('V_max','Trackbars')

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_min,s_min,v_min])

    mask = cv2.inRange(hsv,lower,upper)

    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    frame_contours = frame.copy()

    cv2.drawContours(frame_contours,contours,-1,(0,255,0),3)

    cv2.imshow('webcam',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('contours',frame_contours)

    if cv2.waitkey(1) & 0XFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

