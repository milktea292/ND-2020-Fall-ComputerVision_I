# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2020
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017

import cv2
import numpy as np

cam = cv2.VideoCapture("MarkerCap.mp4")
size = (int(cam.get(3)),int(cam.get(4)))
cam_proc = cv2.VideoWriter('MarkerCap_Proc.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, size)

while (True):
    retval, img = cam.read()

    # res_scale = 0.5 # rescale the input image if it's too large
    # img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)

    # detect selected color (OpenCV uses BGR instead of RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([63,25,51])
    upper = np.array([75,255,230])
    objmask = cv2.inRange(hsv, lower, upper)

    # you may use this for debugging
    cv2.imshow("Binary image", objmask)

    # Resulting binary image may have large number of small objects.
    # You may check different morphological operations to remove these unnecessary
    # elements. You may need to check your ROI defined in step 1 to
    # determine how many pixels your object may have.
    kernel = np.ones((5,5), np.uint8)
    objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
    objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)
    cv2.imshow("Image after morphological operations", objmask)

    # find connected components
    cc = cv2.connectedComponents(objmask)
    ccimg = cc[1].astype(np.uint8)

    # find contours of these objects
    imc, contours, hierarchy = cv2.findContours(ccimg,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

    # You may display the countour points if you want:
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # ignore bounding boxes smaller than "minObjectSize"
    minObjectSize = 15;
    
    for cont in contours:
    
        # use just the first contour to draw a rectangle
        x, y, w, h = cv2.boundingRect(cont)
    
        # do not show very small objects
        if w > minObjectSize or h > minObjectSize:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(img,                        # image
                        "Here's my object!",        # text
                        (x, y-10),                  # start position
                        cv2.FONT_HERSHEY_SIMPLEX,   # font
                        0.7,                        # size
                        (0, 255, 0),                # BGR color
                        1,                          # thickness
                        cv2.LINE_AA)                # type of line

    cv2.imshow("Detection", img)

    cam_proc.write(img)

    action = cv2.waitKey(1)
    if action==27:
        break

cam.release() 
cam_proc.release()
cv2.destroyAllWindows() 