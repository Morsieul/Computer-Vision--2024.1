# Importar as bibliotecas necessárias
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

# Carregar as imagens
image1 = cv2.imread("/home/moe/Documents/images/campus_quixada1.png")
image2 = cv2.imread("/home/moe/Documents/images/campus_quixada2.png")

# Reduz o tamanho das imagens para melhor visualização
h1, w1 = image1.shape[:2]
image1 = cv2.resize(image1, (int(w1*0.5), int(h1*0.5)))

h2, w2 = image2.shape[:2]
image2 = cv2.resize(image2, (int(w2*0.5), int(h2*0.5)) )

# Converter para tons de cinza
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Detectar keypoints e descritores com SIFT
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Matcher BF
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Aplicar o teste de razão
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) >= 4:
    # Extrair localizações dos bons matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Encontrar homografia usando RANSAC
    transformation_matrix, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    
    # Pegar as dimensões das imagens
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Calcular os pontos das extremidades da imagem transformada
    points = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(points, transformation_matrix)
    
    # Combinar com as extremidades da segunda imagem
    points_combined = np.concatenate((transformed_points, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)

    # Encontrar o retângulo delimitador da nova imagem
    [x_min, y_min] = np.int32(points_combined.min(axis=0).ravel())
    [x_max, y_max] = np.int32(points_combined.max(axis=0).ravel())

    # Traduzir a imagem
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])  # Matriz de translação
    
    # Ajustar o tamanho da imagem de saída
    img1_transformed = cv2.warpPerspective(image1, translation_matrix @ transformation_matrix, (x_max - x_min, y_max - y_min))

    # Colocar a segunda imagem sobreposta na primeira
    output_image = img1_transformed.copy()
    output_image[-y_min:h2 - y_min, -x_min:w2 - x_min] = image2

    # Mostrar a imagem combinada
    cv2.imshow("Imagem Combinada", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    raise AssertionError("Não há keypoints suficientes.")
