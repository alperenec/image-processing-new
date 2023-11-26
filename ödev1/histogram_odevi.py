import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"dunyanin-en-uzun-dagi-002.jpg", 0)

# Görüntüyü Matplotlib ile gösterme
plt.imshow(image, cmap='gray')
plt.title('Dag Image')
plt.axis('off')  # Eksenleri kapat
plt.show()

# Histogramı hesaplıyoruz
histogram = {}

for row in image:
    for pixel in row:
        if pixel not in histogram:
            histogram[pixel] = 1
        else:
            histogram[pixel] += 1

# Histogramı Matplotlib ile çizdiriyoruz
plt.bar(histogram.keys(), histogram.values())
plt.xlabel('Piksel Değeri')
plt.ylabel('Frekans')
plt.title('Histogram')
plt.show()
