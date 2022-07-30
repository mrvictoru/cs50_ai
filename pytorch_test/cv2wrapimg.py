# this script should read an image and detect click on the image and warp the image to the clicked point

import cv2
import numpy as np
import sys

# create numpy array to store 4 points
circles = np.zeros((4,2), dtype=np.int32)
# create global variable counter
counter = 0

def click_event(event, x, y, flags, param):
    #access counter global variable
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        #store clicked point in list and store counter
        circles[counter] = x,y
        counter += 1


def main():
    # check argument input using try and except
    if len(sys.argv) != 2:
        print("Usage: cv2wrapimg.py <image>")
        sys.exit(1)
    
    print("Reading image ... {}".format(sys.argv[1]))

    # define width and height of wrap image
    width, height = 250,350
    # read the image
    img = cv2.imread(sys.argv[1])
    # resize img
    img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)

    print ("Press 'Q' to quit")
    while True:
        # check if counter is 4, excute warp function
        if counter == 4:
            # load in point from click event
            pts1 = np.float32(circles)
            pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
            # calculate the perspective transform matrix
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            # warp the image
            wrap = cv2.warpPerspective(img, matrix, (width, height))
            # show the image
            cv2.imshow('Output image', wrap)

        # draw points on the image based on stored points
        for i in range(4):
            cv2.circle(img, (circles[i][0], circles[i][1]), 3, (0,255,0), -1)

        # show the image
        cv2.imshow('Original image', img)
        # set the callback function for mouse event
        cv2.setMouseCallback('Original image', click_event)
        # wait for the user to press q key
        
        if cv2.waitKey(1) == ord('q'):
            # destroy the window
            cv2.destroyAllWindows()
            break
        

        

if __name__ == '__main__':
    main()

