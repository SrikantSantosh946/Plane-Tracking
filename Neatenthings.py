
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
#We want an input video, then select the region of interest, we can then put trackers on this 


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file, make sure it is an mp4")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
    help="OpenCV object tracker type")
ap.add_argument("-o", "--output", type=str,
        help="path to output file")
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
framecount=0
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
initBB = None #Initialise bounding box

def genericvideochange(video_src):#Function that has code contained in both programmes
    fps = None #initialise fps and grab video source from args
    vs = cv2.VideoCapture(video_src)
    while True:
        frame = vs.read()
        frame = frame[1] if video_src == False else frame
        framecount = framecount+1
        if frame is None:
            break       
        (H, W) = frame.shape[:2] #Resize frame but is it necessary to add this?
        frame = imutils.resize(frame, width=600)

def newwindow():#Function to add cutout window
    genericvideochange(args["video"]) #Define video source
    while True:
        if initBB is not None: #find new bounding box co-ordinates
            (success, box) = tracker.update(frame)
            if success: #Draw the bounding box around tracked area and write out this area as a cutout
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
                framegrab = frame[y : y + h , x : x + w ]
                output = cv2.resize(framegrab,(854,480)).astype(np.uint8)
                cv2.line(output,(0,240),(853,240),(255,255,255),1)
                cv2.line(output,(427,0),(427,479),(255,255,255),1)
                out.write(output)
                cv2.imshow("Output", output)
                output = args["output"]
            fps.update()
            fps.stop()

            info = [ #Tell us if successful
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            for (i, (k, v)) in enumerate(info): #Put on information for the tracked area
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"): #Pause then select ROI after pushing 's'
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            tracker.init(frame, initBB)
            fps = FPS().start()
        elif key == ord("q"):
            break
        else:
            vs.release() 
            cv2.destroyAllWindows()
            out.release()               
def MultiTracker():
    trackers = cv2.MultiTracker_create() #multi tracker function
    genericvideochange(args["output"]) #Use the output from the previous function as the input
    while True:
        (success, boxes) = trackers.update(frame)
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"): #seelect the first frame press enter, then press s and then select more frames in this way
            box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
            trackers.add(tracker, frame, box)   
        elif key == ord("q"):
            break  
        else:
            vs.release()
            cv2.destroyAllWindows()
    if __name__ == "__main__":#Master function to execute them
        newwindow()
        MultiTracker()
  



