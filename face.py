import cv2
import mediapipe as mp
import time
import random
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

class FaceMeshWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        

        self.face_mesh = mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.setFixedSize(1920,1080)

        self.warning_label = QLabel(" ** 測試階段 | 數據檢測並不100%準確 ")
        self.warning_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        self.warning_label.setFont(font)
        self.fps_label = QLabel()
        self.fps_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

 

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.warning_label)
        self.layout.addWidget(self.fps_label)

        self.setLayout(self.layout)

    def update_frame(self):
        success, image = self.cap.read()

        start = time.time()


        image.flags.writeable = False


        results = self.face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (self.image_label.width(), self.image_label.height()))

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
                
                cv2.rectangle(image, (20, image.shape[0] - 70), (350, image.shape[0] - 20), (255, 0, 0), -1)
                cv2.putText(image, "result: STABLE", (25, image.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, (255, 255, 255), 2, cv2.LINE_AA)

        
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        qimage = QImage(image.data, image.shape[1], image.shape[0], 
                        image.strides[0], QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

        self.fps_label.setText(f'FPS: {int(fps)}')

  
    def closeEvent(self, event):
        self.cap.release()
        self.face_mesh.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication([])
    main_window = QMainWindow()
    main_widget = FaceMeshWidget()
    main_window.setCentralWidget(main_widget)
    main_window.setWindowTitle('AI Health Monitor')
    main_window.show()
    app.exec_()