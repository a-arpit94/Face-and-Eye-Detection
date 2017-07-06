import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if ret:
        img= cv2.flip(img,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # centre = ((x+(x+w)) /2 ,(y+(y+h))/2)
            # cv2.circle(img,centre ,w/2, (255,0,0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                centre = ((ex + (ex + ew)) / 2, (ey + (ey + eh)) / 2)
                cv2.circle(roi_color,centre ,ew/2, (0,255,0), 2)
                # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('img',img)


    k= cv2.waitKey(30) & 0xff

    # if the 'q' key is pressed, stop the loop
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
