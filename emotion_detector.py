import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break 

    try:
      
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(analysis, list) and len(analysis) > 0:
            emotion = analysis[0]['dominant_emotion']
            
            face_region = analysis[0]['region']
            x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            text = f"Emotion: {emotion.capitalize()}"
            
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        pass

    cv2.imshow('AI Therapy Game - Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Emotion detection module stopped.")