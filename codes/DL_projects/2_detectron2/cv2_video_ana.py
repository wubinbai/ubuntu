import cv2
video = cv2.VideoCapture('video-input.mp4') 
cv2.namedWindow('name',cv2.WINDOW_NORMAL)
cv2.resizeWindow('name',1920,1080) 
while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        break
    #if i % 10 == 0:
    if i > -1:
        FRAME = frame
        print('FRAME....',i)
        cv2.imshow('name',frame)
        cv2.waitKey(500)
    i += 1
