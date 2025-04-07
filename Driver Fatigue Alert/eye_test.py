import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import winsound  # Windows için ses çıkışı

# Yüz tanımlayıcı ve landmark tespit edici yükle
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Parametreler
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0

# Video yakalama başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        leftEye = landmarks[36:42]
        rightEye = landmarks[42:48]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "UYARI! Uyuklama Tespit Edildi!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Ses uyarısı
                winsound.Beep(1000, 500)  # 1000 Hz frekansta 500 ms uzunlukta ses çıkar
        else:
            COUNTER = 0

        # Gözleri çizin
        for (x, y) in leftEye:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
        for (x, y) in rightEye:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
