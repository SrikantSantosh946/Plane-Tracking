#python contour tracking.py --video 'exite --output 'outputfile.dat'
import cv2
import argparse
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
    help="path to input video file")
#ap.add_argument("-o", "--output", type=str, help="path to output file")
args = vars(ap.parse_args())
cap=cv2.VideoCapture(args["video"])
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter("output.mp4", fourcc, 15.0, (1280, 360))

ret, current_frame = cap.read()
ret, previous_frame = cap.read()

if not cap.isOpened():
    print ("Could not open video")

while cap.isOpened():
        diff = cv2.absdiff(current_frame, previous_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5),0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations = 3)
        contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h)= cv2.boundingRect(contour)
            if cv2.contourArea(contour)<10000:
                continue
            cv2.rectangle(current_frame, (x,y), (x+w,y+h), (0,255,0),2)
            cv2.putText(current_frame, 'Status: {}' .format('Movement'),(10,20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    

        image = cv2.resize(current_frame, (1280,720))
        out.write(image)
        cv2.imshow('feed', current_frame)
        current_frame = previous_frame
        ret, previous_frame = cap.read()
    
        if cv2.waitKey(40) == 27:
            break

  
cap.release()
cv2.destroyAllWindows()
out.release()