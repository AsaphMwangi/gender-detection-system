import cvlib as cv
import cv2
import numpy as np

#used adiense data set
#opencv dnn caffe model was used

#this is used to open the webcam
webcam = cv2.VideoCapture(0)
#to check webcam has been successfully launched
if not webcam.isOpened():
    print("Could not open webcam")
    exit()   
     
padding = 20
while webcam.isOpened():
    status, frame = webcam.read()
    face, confidence = cv.detect_face(frame)
    for idx, f in enumerate(face):    
        # get corner points of face rectangle    
        (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
        
        (endX,endY) = min(frame.shape[1]-1, f[2]+padding), min(frame.shape[0]-1, f[3]+padding)
         # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX]) 
        
        (label, confidence) = cv.detect_gender(face_crop)
        
        idx = np.argmax(confidence)
        
        label = label[idx]
        
        label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
        
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)
        
    cv2.imshow("Gender detection System", frame)
    # if you want to stop press "s"
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
webcam.release()
cv2.destroyAllWindows()
