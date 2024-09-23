import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

#abre imagem
filename = sys.argv[1]
img = cv2.imread(filename)

# Pegar o range das cores
azul_baixo = np.array([1, 24, 45])
azul_alto = np.array([135,200,255])

verde_baixo = np.array([60, 110, 60])
verde_alto = np.array([165, 200, 120])


#converte cores
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mascara_azul = cv2.inRange(hsv, azul_baixo, azul_alto)

# mascara_nao_azul = cv2.bitwise_not(mascara_azul)

mascara_verde = cv2.inRange(hsv, verde_baixo, verde_alto)

# mascara_nao_verde = cv2.bitwise_not(mascara_verde)

img[mascara_azul > 0] = [0, 255, 0]
img[mascara_verde > 0] = [255, 0, 0]

plt.subplot(221), plt.imshow(img)
plt.show()





