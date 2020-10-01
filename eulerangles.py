
# python Multitrack.py --video videos/soccer_01.mp4 --tracker csrt

from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
"""Choose left then right wingtip, then pick central fuselage, tail, nose and backlight"""

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
    help="OpenCV object tracker type")
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

framecount = 0
trackers = cv2.MultiTracker_create()
position_3d  = [[] for pos in range(6)]
join = []
f = 300
real_width = 12.5e3  
pixel_perradian = (4.29e-3)/300
dist = []
distance_object = []
clist = [[] for minilist in range(6)]
pitchlist = []
rolllist = []
yawlist = []
# if a video path was not supplied, grab the reference to the web cam
def vector_finder(point1,point2):
    difference = []
    for i in range(3):
        difference.append(point1[i]-point2[i])
    return (difference)

def anglefinder(point1,point2,point3,point4):
    vector_1 = vector_finder(point1,point2)
    vector_2 = vector_finder(point3,point4)
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return math.degrees(angle)

# otherwise, grab a reference to the video file
vs = cv2.VideoCapture(args["video"])

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame = frame[1] if args.get("video", False) else frame
    framecount+=1


    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame, width=1000)
    height = vs.get(3)
    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    lists = [[] for minilist in range(6)]
    # loop over the bounding boxes and draw then on the frame
    for (minilist,box) in zip(lists,boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        minilist.append([(x+(w/2)),(y+(h/2))])
    if lists[0] and lists[1] != []:
        cord = []
        join.append(tuple(minilist for minilist in lists))  
        a=(int(join[-1][0][0][0]),int(join[-1][0][0][1]))
        b = (int(join[-1][1][0][0]),int(join[-1][1][0][1]))
        cv2.line(frame, a, b, [0, 0, 255], 1)
        for i in range(6):
            if lists[i] != []:
                dist.append(math.hypot((b[0]-a[0]),(b[1])-a[1]))
                distance_object.append((real_width/(dist[-1]*pixel_perradian))*1e-6)           
                clist[i].append(lists[i][0])
                X = ((clist[i][-1][0]-clist[i][0][0])*pixel_perradian*distance_object[-1])
                Y = ((560-(clist[i][-1][1]-clist[i][0][1])))*pixel_perradian*distance_object[-1]
                if i == 4:
                    Z = (distance_object[-1]**2) - (X**2) - (Y**2) - 0.01975
                if i == 5:
                    Z = (distance_object[-1]**2) - (X**2) - (Y**2) + 0.01975
                else:
                    Z = (distance_object[-1]**2) - (X**2) - (Y**2)
                position_3d[i] = tuple([X,Y,Z])
                if position_3d[0] and position_3d[1] and position_3d[2] and position_3d[3] and position_3d[4] and position_3d[5] != []:               
                    pt1, pt2, pt3, pt4 = position_3d[0],position_3d[1],position_3d[2],position_3d[3]#XY
                    p1,p2,p3,p4 = position_3d[0],position_3d[1],position_3d[4],position_3d[5]#XZ
                    pts1,pts2,pts3,pts4 = position_3d[2],position_3d[3],position_3d[4],position_3d[5]#YZ
                    rolllist.append(anglefinder(pt1, pt2, pt3, pt4))
                    yawlist.append(anglefinder(p1,p2,p3,p4))
                    pitchlist.append(anglefinder(pts1,pts2,pts3,pts4))


                

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)

        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)

  # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break


else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
time = np.linspace(0,(len(rolllist)/fps),len(rolllist))
def Distance_time_graph(y_coord,y_label):
    plt.plot(time, y_coord)
    plt.xlabel("time (s)")
    plt.ylabel(y_label)
    plt.show()
Distance_time_graph(rolllist, "Roll (Degrees)")
Distance_time_graph(yawlist, "Yaw (Degrees")
Distance_time_graph(pitchlist,"Pitch (Degrees)")