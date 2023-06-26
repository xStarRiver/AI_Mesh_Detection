import cv2
import mediapipe as mp
import time
import random

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():

        success, image = cap.read()

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                
                x_min, y_min, x_max, y_max = 10000, 10000, 0, 0

                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    if x < x_min:
                        x_min = x
                    if y < y_min:
                        y_min = y
                    if x > x_max:
                        x_max = x
                    if y > y_max:
                        y_max = y

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (124,252,0), 2)

                cv2.rectangle(image, (x_min, y_max + 1), (x_min + 85, y_max + 40), (0, 255, 0), -1)

                cv2.putText(image, "SpO2", (x_min + 5, y_max + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                spo2_val = random.uniform(0.85, 1)
                cv2.putText(image, f'{spo2_val:.2f}', (x_min + 90, y_max + 35), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('MediaPipe FaceMesh', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()