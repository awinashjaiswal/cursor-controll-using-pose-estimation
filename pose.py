#!/usr/bin/env python
 
import cv2
import numpy as np
import dlib
import pyautogui
import time
# Read Image
#im = cv2.imread("face.jpg")
#gray=cv2.imread("face.jpg",0)
gray=[]
camera=cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('xmlfile/haarcascade_frontalface_alt.xml')
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

calib_data = np.load('calib.npz')
cmx = calib_data['cmx']
dist = calib_data['dist']
x=y=0
while True:
    ret,frame=camera.read()
    frame=cv2.flip(frame,1)
    #print frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        gray=gray[y:y+h+30,x:x+w]
        newf=frame[y:y+h+30,x:x+w]
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face

        shape = predictor(clahe_image, d) #Get coordinates
        #size = im.shape
             
        #2D image points. If you change the image, you need to change vector
        image_points = np.array([
                                    (shape.part(30).x, shape.part(30).y),     # Nose tip
                                    (shape.part(8).x, shape.part(8).y),     # Chin
                                    (shape.part(36).x, shape.part(36).y),     # Left eye left corner
                                    (shape.part(45).x, shape.part(45).y),     # Right eye right corne
                                    (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
                                    (shape.part(54).x, shape.part(54).y)      # Right mouth corner
                                ], dtype="double")
         
        # 3D model points.
        model_points = np.array([
                                    (0.0, 0.0, 0.0),             # Nose tip
                                    (0.0, -330.0, -65.0),        # Chin
                                    (-225.0, 170.0, -135.0),     # Left eye left corner
                                    (225.0, 170.0, -135.0),      # Right eye right corne
                                    (-150.0, -150.0, -125.0),    # Left Mouth corner
                                    (150.0, -150.0, -125.0)      # Right mouth corner
                                 
                                ])
         
         
        # Camera internals
         
        
        camera_matrix = cmx
         
        #print "Camera Matrix :\n {0}".format(camera_matrix)
         
        #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist)
         
        #print "Rotation Vector:\n {0}".format(rotation_vector)
        #print "Translation Vector:\n {0}".format(translation_vector)
         
         
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
         
         
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist)
         
        for p in image_points:
            cv2.circle(newf, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
         
         
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        #pyautogui.moveTo(int(nose_end_point2D[0][0][0]),int(nose_end_point2D[0][0][1]), duration=".001")
        cv2.line(newf, p1, p2, (255,0,0), 2)
        x_axis=400+x+shape.part(30).x+p2[0]
        y_axis=y+shape.part(30).y+p2[1]
        right_eyeblink=shape.part(47).y-shape.part(44).y #---- for rigt eye
        left_eyeblink=shape.part(40).y-shape.part(37).y #---- for rigt eye
        # --------------- controll the cursor ------------------------#
        pyautogui.moveTo(int(x_axis),int(y_axis), duration=0.0001)
        print left_eyeblink
        if right_eyeblink<=4 and left_eyeblink>3:          #---- check for right eye click ------#
            print "right click"
            pyautogui.click(button='right')
        if left_eyeblink<=5 and right_eyeblink>4:
            pyautogui.click(button='left')
        #print nose_end_point2D 
        # Display image
        cv2.imshow("Output", frame)
    if cv2.waitKey(3) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
