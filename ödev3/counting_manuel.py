import cv2
import numpy as np

def count_rice(image):
    rice_count = 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) 

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100:
            rice_count += 1

    return rice_count

file_path = 'pirincfotosu.jpg'

def apply_threshold(file_path, threshold):
    image = cv2.imread(file_path)

    #eşikleme ile daha net bir görüntüde saymak kolay oldu
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return thresholded_image

threshold_value = 200

thresholded_image = apply_threshold(file_path, threshold_value)

total_rice_count = count_rice(thresholded_image)

#ekranda da sayıyı veriyor
cv2.putText(thresholded_image, f'Rice Count: {total_rice_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Thresholded Image', thresholded_image)
cv2.imwrite("yedek.jpg", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(total_rice_count)            #konsolda da yazdırıyor


