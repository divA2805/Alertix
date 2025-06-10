import cv2
import numpy as np
import dlib
from imutils import face_utils
from pygame import mixer

mixer.init()
mixer.music.load(r"C:\Users\Ashutosh Sharma\Final\alarm.wav")

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Ashutosh Sharma\shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
alarm_time = None

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(
            landmarks[36],
            landmarks[37],
            landmarks[38],
            landmarks[41],
            landmarks[40],
            landmarks[39],
        )
        right_blink = blinked(
            landmarks[42],
            landmarks[43],
            landmarks[44],
            landmarks[47],
            landmarks[46],
            landmarks[45],
        )

        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 5:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                if alarm_time is None:
                    alarm_time = cv2.getTickCount()  # Start the alarm timer
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 5:
                status = "Drowsy !"
                color = (0, 0, 255)
                if alarm_time is None:
                    alarm_time = cv2.getTickCount()  # Start the alarm timer
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 5:
                status = "Active :)"
                color = (0, 255, 0)
                alarm_time = None  # Reset the alarm timer

        # Check if it's time to play the alarm
        if alarm_time is not None:
            current_time = cv2.getTickCount()
            time_diff = (current_time - alarm_time) / cv2.getTickFrequency()
            if time_diff >= 5:
                mixer.music.play()
                alarm_time = None  # Reset the alarm timer

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
        
        cv2.imshow("Result of detector", face_frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()