import cv2 as cv
import dlib
from beepy import beep
from scipy.spatial import distance

# Calculate the Eye Aspect Ratio (EAR) using specific eye landmarks
def get_eye_aspect_ratio(eye_points):
    dist_A = distance.euclidean(eye_points[1], eye_points[5])
    dist_B = distance.euclidean(eye_points[2], eye_points[4])
    dist_C = distance.euclidean(eye_points[0], eye_points[3])
    ear_value = (dist_A + dist_B) / (2.0 * dist_C)
    return ear_value

video_capture = cv.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame_counter = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detected_faces = face_detector(gray_frame)
    for face in detected_faces:

        face_landmarks = landmark_predictor(gray_frame, face)
        left_eye_points = []
        right_eye_points = []

        for idx in range(36, 42):
            x = face_landmarks.part(idx).x
            y = face_landmarks.part(idx).y
            left_eye_points.append((x, y))
            next_idx = idx + 1
            if idx == 41:
                next_idx = 36
            x2 = face_landmarks.part(next_idx).x
            y2 = face_landmarks.part(next_idx).y
            cv.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for idx in range(42, 48):
            x = face_landmarks.part(idx).x
            y = face_landmarks.part(idx).y
            right_eye_points.append((x, y))
            next_idx = idx + 1
            if idx == 47:
                next_idx = 42
            x2 = face_landmarks.part(next_idx).x
            y2 = face_landmarks.part(next_idx).y
            cv.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = get_eye_aspect_ratio(left_eye_points)
        right_ear = get_eye_aspect_ratio(right_eye_points)

        avg_ear = (left_ear + right_ear) / 2.0
        avg_ear = round(avg_ear, 2)
        if avg_ear <= 0.20:
            frame_counter += 1
            if frame_counter == 10:
                cv.putText(frame, "Drowsiness Alert", (20, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                cv.putText(frame, "Please Wake Up", (20, 100),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                beep(sound=3)
                print("Drowsiness detected")
        else:
            frame_counter = 0
        print(avg_ear)

    cv.imshow("Alertness sensor", frame)

    key_press = cv.waitKey(1)
    if key_press == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv.destroyAllWindows()
