
# python newwindowtrack.py --video movie.mp4  --outputfile outpufile.avi
#Make sure the output is an .avi file

from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np

framecount = 0


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
    help="OpenCV object tracker type")
ap.add_argument("-o", "--output", type=str,
        help="path to output file")
args = vars(ap.parse_args())
fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
out = cv2.VideoWriter(args["output"], fourcc, 10, (854,480))


OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# open outputfile for writing
#f = open(args["outputfile"],"w+")
# initialize the bounding box coordinates of the object we are going
# to track
initBB = None
vs = cv2.VideoCapture(args["video"])
# initialize the FPS throughput estimator
fps = None
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    framecount = framecount+1
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensionss
#   frame = imutils.resize(frame, width=1000)
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
#            f.write("%d %g %g %g %g\n" % (framecount, x, w, y, h))
            framegrab = frame[y : y + h , x : x + w ]
            output = cv2.resize(framegrab,(854,480)).astype(np.uint8)
            cv2.line(output,(0,240),(853,240),(255,255,255),1)
            cv2.line(output,(427,0),(427,479),(255,255,255),1)
            out.write(output)
            cv2.imshow("Output", output)
        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
#    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break


else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
out.release()