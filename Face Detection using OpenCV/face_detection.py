import cv2

def detect():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load face cascade.")
        return

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    if eye_cascade.empty():
        print("Warning: Could not load eye cascade. Eye detection will be skipped.")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex + x, ey + y), (ex + ew + x, ey + eh + y), (0, 255, 0), 2)

        cv2.imshow('camera', frame)  

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()