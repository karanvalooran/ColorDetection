#importing libraries
from cv2 import cv2
import imutils
import numpy as np

#Input image
frame = cv2.imread("Rubiks Cube.jpg")
#Obtaining the width and height of the frame
(x,y) = frame.shape[:2] 
#Copying the frames
redframe = frame.copy()
blueframe = frame.copy()
greenframe = frame.copy()
yellowframe = frame.copy()

#converting the frame from BGR to HSV (For Color detection)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#HSV value ranges for different colors
    #Green 
lower_green = np.array([40,70,80])
upper_green = np.array([70,255,255])
    #Blue
lower_blue = np.array([90,60,0])
upper_blue = np.array([121,255,255])
    #lower red
llow_red = np.array([0,50,70])
lhigh_red = np.array([9, 255, 255])
    #upper red
ulow_red = np.array([159,50,70])
uhigh_red = np.array([180, 255, 255])

        
#Creating masks for different colors
mask1 = cv2.inRange(hsv, lower_green, upper_green)
lred_mask = cv2.inRange(hsv, llow_red, lhigh_red)
ured_mask = cv2.inRange(hsv, ulow_red, uhigh_red)
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
    cv2.drawContours(greenframe, [c], -1, (0,255,0),15) 
                
for c in cnts2:
  area2 = cv2.contourArea(c)
  if area2> 5000:
    cv2.drawContours(redframe, [c], -1, (0,0,255),15)
                
for c in cnts3:
  area3 = cv2.contourArea(c)
  if area3> 5000:
    cv2.drawContours(blueframe, [c], -1, (255,0,0),15)       
      
#Adding text to the Windows
cv2.putText(greenframe,"Green", (int(y/2), int(x/8)), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,255,0), 10) 
cv2.putText(redframe,"Red", (int(y/2), int(x/8)), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 10)
cv2.putText(blueframe,"Blue", (int(y/2), int(x/8)), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,0,0), 10)

#Stacking images in a single window 
imh1 = cv2.hconcat([frame, blueframe])
imh2 = cv2.hconcat([greenframe, redframe])
imv1 = cv2.vconcat([imh1, imh2])

#Saving a copy of the output
cv2.imwrite("Color Detection.jpg", imv1)

#Resizing the window
imv1 = cv2.resize(imv1, (768,768))
cv2.imshow("Color Detection",imv1)
key = cv2.waitKey(0)
if key ==27:
  cv2.destroyAllWindows()
