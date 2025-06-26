import cv2

harcascade = "model/haarcascade_frontalface_default (1).xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# Load Haar Cascade only once (outside the loop)
facecascade = cv2.CascadeClassifier(harcascade)

while True:
    success, img = cap.read()
    if not success:
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # You can detect faces here if needed, e.g.:
    # faces = facecascade.detectMultiScale(img_gray, 1.1, 4)

    face = facecascade.detectMultiScale(img_gray, 1.1,4)

    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Face", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # ✅ Fixed indentation

cap.release()
cv2.destroyAllWindows()
