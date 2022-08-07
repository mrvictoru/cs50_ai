# object detection using contoursdetection and display using stackimages

import cv2
import numpy as np
from stackimages import stackImages
from contoursdetection import getContours

def empty(a):
    pass

def main():
    # create video capture object
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    
    # create Trackbar for Thrshold1 and Threshold2
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 800, 240)
    cv2.createTrackbar("Threshold1", "Parameters", 0, 255, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 0, 255, empty)
    
    # read and display frame, press q to quit
    while True:
        success, img = cap.read()

        # convert to grayscale
        imgBlur = cv2.GaussianBlur(img, (7,7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        
        # use threadhold for canny edge detection
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        

        # create stack of images and display it
        imgStack = stackImages(0.8, ([img, imgGray]))
        cv2.imshow("Stack", imgStack)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # release video capture object
    cap.release()
    


if __name__ == "__main__":
    main()

