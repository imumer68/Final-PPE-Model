import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    img = cv2.Canny(frame, 100, 200)  # Some image processing
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
