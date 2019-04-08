from __future__ import division
import cv2
from imutils import face_utils
import time
import imutils
import dlib
import numpy as np

import headPose
import yawning
import blinking
import arduino
import ann

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def drowsinessDetection():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    # head pose estimation
    pitch_thresh = 10
    slump_counter = 0
    slump_total = 0
    temp = False
    
    # yawning detection
    yawns = 0
    LIP_DISTANCE = 25
    yawn_total = 0
    
    # blink detection
    ear_thresh = 0.2
    counter = 0 
    blink_num = 0
    blink_duration_sum = 0
    (left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return
    
    startTime = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        frame_counter += 1
        if ret == True:
            frame = imutils.resize(frame, width=450, height=350)

            # head pose estimation
            face_rects = detector(frame, 0)
            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                reprojectdst, euler_angle = headPose.get_head_pose(shape)
                
                try:
                    for start, end in headPose.line_pairs:
                        cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                except:
                    break

                pitch = euler_angle[0, 0]
                #yaw = euler_angle[1, 0]
                #roll = euler_angle[2, 0]
                
                if pitch > pitch_thresh:
                    slump_counter += 1
                    temp = True
                else:
                    if slump_counter > 0:
                        slump_total += 1
                    temp = False
                    slump_counter = 0

                cv2.putText(frame, "Slumps per sec: " + "{:7.2f}".format(slump_total / (time.time() - startTime)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), thickness=2)
                
            # yawning detection
            image_landmarks, lip_distance = yawning.mouth_open(frame)
            if lip_distance > LIP_DISTANCE:
                yawns += 1
            else:
                if yawns > 0:
                    yawn_total += 1
                yawns = 0
                
            cv2.putText(frame, "Yawns per sec: {}".format(yawn_total / (time.time() - startTime)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            # blink detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[left_start:left_end]
                right_eye = shape[right_start:right_end]

                leftEAR = blinking.eye_aspect_ratio(left_eye)
                rightEAR = blinking.eye_aspect_ratio(right_eye)
                EAR = (leftEAR + rightEAR) / 2.0

                blinking.visualize_eyes(left_eye, right_eye, frame)

                if EAR < ear_thresh:
                    counter += 1
                else:
                    if counter > 0:
                        blink_num += 1 
                        blink_duration_sum += counter
                    counter = 0

                cv2.putText(frame, "Blinks per sec: {}".format(blink_num / (time.time() - startTime)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if blink_num != 0:
                    cv2.putText(frame, "Average blinking frame num: {}".format(blink_duration_sum / blink_num), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
            if frame_counter > 100:
                # all
                a = slump_total / (time.time() - startTime)
                b = yawn_total / (time.time() - startTime)
                c = blink_num / (time.time() - startTime)
                if blink_num > 0:
                    d = blink_duration_sum / blink_num
                else: d = 0

                #detecting drowsines
                result = round(ann.predict(np.array([[a, b, c, d]])))
                print(result)
                if result == 0.0: signal = '0'
                else: signal = '1'
                print('signal: ' + signal)
                arduino.send_signal(signal)
                frame_counter = 0
            
            cv2.imshow("Frame", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        else:
            break
    cap.release()           
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    drowsinessDetection()