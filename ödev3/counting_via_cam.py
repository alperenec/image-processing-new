import cv2
import numpy as np

def apply_threshold(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return thresholded_image


def count_rice(image):
    rice_count = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Eşikleme ile net görüntü
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100:
            rice_count += 1

    return rice_count


cap = cv2.VideoCapture(0)

threshold_value = 200

while True:
    ret, frame = cap.read()

    thresholded_frame = apply_threshold(frame, threshold_value)

    total_rice_count = count_rice(frame)

    #ekranda da sayıyı veriyor
    cv2.putText(frame, f'Rice Count: {total_rice_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)

    #fotograf alırken q tuşluna basmak işimi kolaylastırdı
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.imshow('Thresholded Image', thresholded_frame)
cv2.imwrite("yedek3.jpg", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(total_rice_count)  #konsolda da çıktı veriyor