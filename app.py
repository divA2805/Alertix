from flask import Flask, render_template, Response
import cv2
import numpy as np
import dlib
from imutils import face_utils
from pygame import mixer

app = Flask(__name__)

mixer.init()
mixer.music.load(r"C:\Users\Ashutosh Sharma\Final\alarm.wav")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r"C:\Users\Ashutosh Sharma\shape_predictor_68_face_landmarks.dat"
)

def detect_faces():
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
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera")
            break
        
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

            for n in range(0, 68):
                (x, y) = landmarks[n]
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

            cv2.putText(
                frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
            )

        _, jpeg = cv2.imencode(".jpg", frame)
        frame_bytes = jpeg.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()

def detect_poses_openpose():
    net = cv2.dnn.readNetFromTensorflow(r"C:\Users\Ashutosh Sharma\graph_opt.pb")
    threshold = 0.2
    width = 368
    height = 368

    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

    inWidth = width
    inHeight = height
    cap = cv2.VideoCapture(0)  # Open the default camera (usually the webcam)
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.waitKey()
            break

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponding body part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if its confidence is higher than the threshold.
            points.append((int(x), int(y)) if conf > threshold else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        _, jpeg = cv2.imencode(".jpg", frame)
        frame_bytes = jpeg.tobytes()
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/opencvDlib2")
def opencv():
    return render_template("opencvDlib.html")

@app.route("/openpose")
def openpose():
    return render_template("openpose.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        detect_faces(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_feed_openpose")
def video_feed_openpose():
    return Response(
        detect_poses_openpose(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/tinker_video")
def tinkercad_video():
    return render_template("tinker_video.html")


if __name__ == "__main__":
    app.run(debug=True)