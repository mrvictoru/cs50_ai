# object detection using detrdetect and display using stackimages

import cv2
import numpy as np
from stackimages import stackImages
from detrdetect import getobject

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
    cv2.createTrackbar("Threshold1", "Parameters", 0.0001, 0.9999, empty)

    
    # read and display frame, press q to quit
    while True:
        success, img = cap.read()
        # use threadhold for model threshold
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")

        # use getobject to get bounding box and display them on the original image
        imgDetect = getobject(img, threshold1)
        
        # create stack of images and display it
        imgStack = stackImages(0.8, ([img, imgDetect]))
        cv2.imshow("Stack", imgStack)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # release video capture object
    cap.release()
    


if __name__ == "__main__":
    main()

