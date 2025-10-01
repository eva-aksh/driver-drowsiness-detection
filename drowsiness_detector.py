from playsound import playsound
import time
import cv2
import mediapipe as mp 

# Initialize medimport cv2
import mediapipe as mp
from mediapipe import solutions
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indexes (left eye upper & lower lid)
LEFT_EYE = [159, 145]
RIGHT_EYE = [386, 374]

# Drowsiness detection variables
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 15

# Camera open 
cap = cv2.VideoCapture(0)

def get_eye_openness(landmarks, eye_indices, image_shape):
    y1 = int(landmarks[eye_indices[0]].y * image_shape[0])
    y2 = int(landmarks[eye_indices[1]].y * image_shape[0])
    distance = abs(y2 - y1)
    return distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_eye_openness = get_eye_openness(face_landmarks.landmark, LEFT_EYE, frame.shape)
            right_eye_openness = get_eye_openness(face_landmarks.landmark, RIGHT_EYE, frame.shape)
            
            avg_openness = (left_eye_openness + right_eye_openness) / 2

            if avg_openness < 5:  
                closed_frames += 1
            else:
                closed_frames = 0

            if closed_frames >= CLOSED_FRAMES_THRESHOLD:
                print("driver is sleeping")
                playsound('\a')  
                closed_frames = 0

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
