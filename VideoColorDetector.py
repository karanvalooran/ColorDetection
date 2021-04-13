#Color Detection using OpenCV - Python

#Importing the libraries 
from cv2 import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture("test.mp4")

while True:
    #Extracting the frames
    ret, frame = cap.read()
    if ret is True:

        #Resizing the frame
        frame = cv2.resize(frame, (320,320))

        #Copying the frames
        redframe = frame.copy()
        blueframe = frame.copy()
        greenframe = frame.copy()
        yellowframe = frame.copy()

        #converting the frame from BGR to HSV (For Color detection)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #HSV value ranges for different colors
        lower_green = np.array([40,70,80])
        upper_green = np.array([70,255,255])

        lower_blue = np.array([90,60,0])
        upper_blue = np.array([121,255,255])
        
        #lower red
        llow_red = np.array([0,50,70])
        lhigh_red = np.array([9, 255, 255])
        lred_mask = cv2.inRange(hsv, llow_red, lhigh_red)
        #upper red
        ulow_red = np.array([159,50,70])
        uhigh_red = np.array([180, 255, 255])
        ured_mask = cv2.inRange(hsv, ulow_red, uhigh_red)
        
        #Creating masks for different colors
        mask1 = cv2.inRange(hsv, lower_green, upper_green)
        mask2 = lred_mask+ured_mask
        mask3 = cv2.inRange(hsv, lower_blue, upper_blue)

        #Creating contours or boundaries for each colors
        cnts1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = imutils.grab_contours(cnts1)
        
        cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)

        cnts3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts3 = imutils.grab_contours(cnts3)

        for c in cnts1:
            area1 = cv2.contourArea(c)
            if area1> 5000:
                cv2.drawContours(greenframe, [c], -1, (0,255,0),3) 
                
        for c in cnts2:
            area2 = cv2.contourArea(c)
            if area2> 5000:
                cv2.drawContours(redframe, [c], -1, (0,0,255),3)
                
        for c in cnts3:
            area3 = cv2.contourArea(c)
            if area3> 5000:
                cv2.drawContours(blueframe, [c], -1, (255,0,0),3)       
      

        cv2.putText(greenframe,"Green", (120, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1) 
        cv2.putText(redframe,"Red", (120, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
        cv2.putText(blueframe,"Blue", (120, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        imh1 = cv2.hconcat([frame, blueframe])
        imh2 = cv2.hconcat([greenframe, redframe])
        imv1 = cv2.vconcat([imh1, imh2])
        cv2.imshow("Color Detection", imv1)
        key=cv2.waitKey(30)
        if key==27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
