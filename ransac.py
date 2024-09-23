import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# Carregar a imagem
imagem = cv2.imread('/home/moe/Documents/images/pontos_ransac.png')
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Definir intervalo para a cor verde (ajuste conforme necessário)
verde_baixo = np.array([40, 40, 40])
verde_alto = np.array([70, 255, 255])

# Criar máscara para o intervalo da cor verde
mascara_verde = cv2.inRange(hsv, verde_baixo, verde_alto)

# Operações morfológicas para limpar ruídos
kernel = np.ones((5, 5), np.uint8)
mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)
mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)

# Encontrar contornos dos pontos verdes
contornos, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Verificar se foram encontrados contornos
if len(contornos) == 0:
    print("Nenhum ponto verde encontrado.")
else:
    # Extrair coordenadas dos pontos (x, y)
    pontos = []
    for contorno in contornos:
        for ponto in contorno:
            x, y = ponto[0]  # Garantir que cada ponto é uma tupla (x, y)
            pontos.append([x, y])
    
    # Converter lista de pontos para array numpy
    pontos = np.array(pontos)

    # Separar as coordenadas x e y
    X = pontos[:, 0].reshape(-1, 1)  # Coordenadas x
    y = pontos[:, 1]                 # Coordenadas y

    # Verificar se os pontos foram corretamente extraídos
    print(f"Pontos extraídos: {pontos.shape[0]}")

    # Se houver pontos suficientes, aplicar RANSAC
    if pontos.shape[0] > 1:
        # Aplicar RANSAC
        modelo_ransac = RANSACRegressor(estimator=LinearRegression(), 
                                        min_samples=2, 
                                        residual_threshold=5, 
                                        random_state=42)
        modelo_ransac.fit(X, y)
        
        # Obter os parâmetros da reta
        a = modelo_ransac.estimator_.coef_[0]
        b = modelo_ransac.estimator_.intercept_
        print(f"Equação da reta: y = {a:.2f}x + {b:.2f}")

        # Gerar pontos para a reta
        x_min = X.min()
        x_max = X.max()
        y_min = a * x_min + b
        y_max = a * x_max + b

        # Desenhar a reta na imagem original
        imagem_com_reta = imagem.copy()
        cv2.line(imagem_com_reta, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        # Exibir a imagem com a reta ajustada
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(imagem_com_reta, cv2.COLOR_BGR2RGB))
        plt.title('Reta Ajustada com RANSAC')
        plt.axis('off')
        plt.show()
    else:
        print("Pontos insuficientes para ajustar a reta.")
