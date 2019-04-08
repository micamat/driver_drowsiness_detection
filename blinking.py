import cv2
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    
    EAR = (a + b) / (2.0 * c)
    return EAR

def visualize_eyes(left_eye, right_eye, frame):
    left_hull = cv2.convexHull(left_eye)
    right_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [left_hull], -1, (255,0,0), 1)
    cv2.drawContours(frame, [right_hull], -1, (255,0,0), 1)