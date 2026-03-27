import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import dlib
from scipy.spatial import distance as dist
import pygame
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Constants
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
OCR_INTERVAL = 30

# Drowsiness detection
COUNTER = 0
ALARM_ON = False

# Initialize pygame mixer
pygame.mixer.init()

ALERT_SOUND = r"C:\Users\Gokul\Downloads\beep.mp3"
predictor_path = r"C:\Users\Gokul\Downloads\shape_predictor_68_face_landmarks.dat"

# Dlib setup
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Drawing setup
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
blue_index = green_index = red_index = yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255

# Toolbar
buttons = [("CLEAR", (40, 1, 140, 65), (0, 0, 0)),
           ("BLUE", (160, 1, 255, 65), (255, 0, 0)),
           ("GREEN", (275, 1, 370, 65), (0, 255, 0)),
           ("RED", (390, 1, 485, 65), (0, 0, 255)),
           ("YELLOW", (505, 1, 600, 65), (0, 255, 255))]

for text, (x1, y1, x2, y2), color in buttons:
    cv2.rectangle(paintWindow, (x1, y1), (x2, y2), color, 2)
    cv2.putText(paintWindow, text, (x1 + 10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Mediapipe
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

ocr_counter = 0
ocr_result = None
prev_time = 0

with mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        current_time = time.time()
        if current_time - prev_time < 1.0 / 15:
            continue
        prev_time = current_time

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        rects = detector(small_gray)

        for rect in rects:
            rect = dlib.rectangle(left=int(rect.left() * 2),
                                  top=int(rect.top() * 2),
                                  right=int(rect.right() * 2),
                                  bottom=int(rect.bottom() * 2))

            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            ear = (eye_aspect_ratio(leftEye) +
                   eye_aspect_ratio(rightEye)) / 2.0

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (150, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 3)

                    if not ALARM_ON:
                        pygame.mixer.music.load(ALERT_SOUND)
                        pygame.mixer.music.play(-1)
                        ALARM_ON = True
            else:
                COUNTER = 0
                ALARM_ON = False
                pygame.mixer.music.stop()

        # Toolbar
        for text, (x1, y1, x2, y2), color in buttons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1 + 10, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for handlms in result.multi_hand_landmarks:
                lmList = [(int(lm.x * 640), int(lm.y * 480))
                          for lm in handlms.landmark]

                mpDraw.draw_landmarks(frame, handlms,
                                      mpHands.HAND_CONNECTIONS)

                fore_finger = lmList[8]
                thumb = lmList[4]
                center = fore_finger

                cv2.circle(frame, center, 5, (0, 255, 0), -1)

                if thumb[1] - center[1] < 30:
                    bpoints.append(deque(maxlen=512)); blue_index += 1
                    gpoints.append(deque(maxlen=512)); green_index += 1
                    rpoints.append(deque(maxlen=512)); red_index += 1
                    ypoints.append(deque(maxlen=512)); yellow_index += 1

                elif center[1] <= 65:
                    if 40 <= center[0] <= 140:
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]
                        blue_index = green_index = red_index = yellow_index = 0
                        paintWindow[67:, :, :] = 255

                    elif 160 <= center[0] <= 255:
                        colorIndex = 0
                    elif 275 <= center[0] <= 370:
                        colorIndex = 1
                    elif 390 <= center[0] <= 485:
                        colorIndex = 2
                    elif 505 <= center[0] <= 600:
                        colorIndex = 3

                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)

        # Drawing logic (unchanged)
        points = [bpoints, gpoints, rpoints, ypoints]
        for i, pointGroup in enumerate(points):
            for j in range(len(pointGroup)):
                for k in range(1, len(pointGroup[j])):
                    if pointGroup[j][k - 1] and pointGroup[j][k]:
                        cv2.line(frame, pointGroup[j][k - 1],
                                 pointGroup[j][k], colors[i], 2)
                        cv2.line(paintWindow, pointGroup[j][k - 1],
                                 pointGroup[j][k], colors[i], 2)

        # OCR
        if ocr_counter % OCR_INTERVAL == 0:
            gray_canvas = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_canvas, 150, 255, cv2.THRESH_BINARY)

            text = pytesseract.image_to_string(thresh)
            ocr_result = None
            if text.strip():
                try:
                    ocr_result = eval(text.strip())
                except:
                    pass

        ocr_counter += 1

        if ocr_result is not None:
            cv2.putText(frame, f"Result: {ocr_result}",
                        (150, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()