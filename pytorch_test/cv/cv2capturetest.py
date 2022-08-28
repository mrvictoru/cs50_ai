# to test opencv functionaility in python
import numpy as np
import cv2

# create video capture object
cap = cv2.VideoCapture(0)

# read and display frame, press q to quit
while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    print(width, height)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

#release video capture object
cap.release()
cv2.destroyAllWindows()