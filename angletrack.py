#python angletrack.py --video 'exeoutput.avi'
import cv2
import argparse
import math
import imutils
import time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str,
default="csrt", help = "tracker type")
args = vars(ap.parse_args())
OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

trackers = cv2.MultiTracker_create() #Initialise multitracker
vs = cv2.VideoCapture(args["video"]) #Initialise video #Usful for our time axis
anglelist=[] #Empty list for angle computation#Bookkeeping for path tracking and direction of travel
framecount = 0
(dX, dY) = (0, 0)
direction = ""
 
 
def gradient(pt1,pt2):#Simple gradient calculation 
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
 
def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]#Last 3 points detected in the frame
    m1 = gradient(pt1,pt2)
    m2 = gradient(pt1,pt3)
    angR = np.arctan((m2-m1)/(1+(m2*m1)))#Arctan formula for angle using gradients
    angD = round(math.degrees(angR)) #Round to nearest whole number
    cv2.putText(frame,str(angD),(int(pt1[0]-40),int(pt1[1]-20)),cv2.FONT_HERSHEY_COMPLEX,
                1.5,(0,0,255),2) #Put the angle on the frame for the user
 
while True: #Loops indefinitely
    frame = vs.read()#Read video
    frame = frame[1] if args.get("video", False) else frame #Start at beginning or continue to next frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=600)#Resizes the frame so its easier to handle
    (success, boxes) = trackers.update(frame)#grabs the updated bounding box coordinates for ech object
    lists = [[] for _ in range(3)]#create a list of lists that is the same length as the number of tracking boxes
    
   #For every new box a new list of coordinates needs to be tracked 
    for (box,_) in zip(boxes,lists): #Loops over the bounding boxes and draws over them
        (x, y, w, h) = [int(v) for v in box ]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)     
        _.append([x+w/2,y+h/2]) #Add each co-ordinate
        #Middle point of rectangle taken for tracked path

        
        if len(_)<10:
            continue    
        
        if len(_) > 10:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = list1[-10][0] - list1[i][0]
            dY = list1[-10][1] - list1[i][1]
            (dirX, dirY) = ("", "")

                # ensure there is significant movement in the
                # x-direction
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"

                # ensure there is significant movement in the
                # y-direction
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"

                # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)

                # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY  
                
                #thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, tuple(_[i - 1]), tuple(_[i]), (0, 0, 255), 1)
    if lists[0] and lists[1] and lists[2] != []:#When 3 boxes are found we can find the angle
        anglelist.append(tuple(_[-1] for _ in lists))
        a=(int(anglelist[-1][0][0]),int(anglelist[-1][0][1]))
        b = (int(anglelist[-1][1][0]),int(anglelist[-1][1][1]))
        c = (int(anglelist[-1][2][0]),int(anglelist[-1][2][1]))#Add the last value of each list into one list
        cv2.line(frame,a,b,(0,255,0),1)#Lines showing the angle produced
        cv2.line(frame,b,c,(0,255,0),1)
        getAngle(anglelist[-1])
                    
    cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 0, 255), 3)
    cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)    

    cv2.imshow("Frame", frame) #Show the output frame
    key = cv2.waitKey(1) & 0xFF
    framecount +=1
    if key == ord("s"):#push "s" to select a new frame to track
        box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box) #Create new tracker for our bounding box and add to our multi object tracker
   
 
    if key == ord('q'):
        break
else:
    vs.release()
cv2.destroyAllWindows()