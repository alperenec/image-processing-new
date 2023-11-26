import cv2
import matplotlib.pyplot as plt

def gri_histogram(resim):

  # Resmin boyutunu al.
  genişlik, yükseklik = resim.shape

  # Histogramı saklamak için bir dizi oluştur.
  histogram = [0] * 256

  # Resmin her pikseli için histogramı güncelle.
  for i in range(genişlik):
    for j in range(yükseklik):
      histogram[resim[i, j]] += 1

  # Histogramı döndür.
  return histogram

def main():
  # Resmi yükle.
  resim = cv2.imread("resim.jpg", cv2.IMREAD_GRAYSCALE)

  # Histogramı hesapla.
  histogram = gri_histogram(resim)

  # Histogramı görselleştir.
  plt.plot(histogram)
  plt.show()

if __name__ == "__main__":
  main()