import cv2
import numpy as np

def crop_image(img_path, title):

    img = cv2.imread(img_path)
        ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

        ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    cv2.imwrite(  title , dst)
    

crop_image('/home/moe/Documents/images/circle.jpg', 'cabeca.png')
crop_image('/home/moe/Documents/images/line.jpg', 'palito.png')

b_altura = 300
b_largura = 300

background =  np.zeros((b_altura, b_largura, 3), dtype=np.uint8)
background.fill(255)

cabeca = cv2.imread('cabeca.png')
palito = cv2.imread('palito.png')

colagem = background.copy()

x_offset = (b_largura - cabeca.shape[1]) // 2
y_offset = (b_altura  - cabeca.shape[0]) // 2 - 90

colagem[y_offset: y_offset + cabeca.shape[0], x_offset: x_offset + cabeca.shape[1] ] = cabeca

tronco = palito.copy()

tronco_x_offset = (b_largura - palito.shape[1]) // 2 + 36
tronco_y_offset = (b_altura  - palito.shape[0]) // 2 - 80

matriz_rotacao = cv2.getRotationMatrix2D((tronco.shape[1] / 2, tronco.shape[0] / 2), 90 , 1.0)
tronco_rotacionado = cv2.warpAffine(tronco, matriz_rotacao, (tronco.shape[0], tronco.shape[1]))

colagem[tronco_y_offset: tronco_y_offset + tronco_rotacionado.shape[0], tronco_x_offset: tronco_x_offset + tronco_rotacionado.shape[1] ] = tronco_rotacionado

braco_direito = palito.copy()
braco_esquerdo = palito.copy()

braco_altura = int(palito.shape[0] * 0.75)
braco_largura = int(palito.shape[1] * 0.75)

braco_direito = cv2.resize(braco_direito, (braco_largura, braco_altura))
braco_esquerdo = cv2.resize(braco_esquerdo, (braco_largura, braco_altura))

braco_direito_x_offset = (b_largura - palito.shape[1]) // 2 - 20
braco_direito_y_offset = (b_altura  - palito.shape[0]) // 2 + 3

braco_esquerdo_x_offset = (b_largura - palito.shape[1]) // 2 - 20
braco_esquerdo_y_offset = (b_altura  - palito.shape[0]) // 2 - 54
colagem[braco_direito_x_offset: braco_direito_x_offset + braco_direito.shape[0], braco_direito_y_offset: braco_direito_y_offset + braco_direito.shape[1] ] = braco_direito
colagem[braco_esquerdo_x_offset: braco_esquerdo_x_offset + braco_esquerdo.shape[0], braco_esquerdo_y_offset: braco_esquerdo_y_offset + braco_esquerdo.shape[1] ] = braco_esquerdo

perna_direita = palito.copy()
perna_esquerda = palito.copy()

perna_largura = int(braco_largura * 2)
perna_altura = int(braco_altura)

perna_direita = cv2.resize(perna_direita, (perna_largura, perna_altura))
perna_esquerda = cv2.resize(perna_esquerda, (perna_largura, perna_altura))

(h, w) = background.shape[:2]   
(cx, cy) = (w // 2.5, h // 2.5)

rotacao_perna_direita = cv2.getRotationMatrix2D((cx, cy), 90 , 1.0)
perna_direita_rot = cv2.warpAffine(perna_direita, rotacao_perna_direita, (perna_direita.shape[0], perna_direita.shape[1]))
# rotacao_perna_esquerda = cv2.getRotationMatrix2D((cx, cy), 45 , 1.0)
# perna_esquerda_rot = cv2.warpAffine(perna_esquerda, rotacao_perna_esquerda, (perna_esquerda.shape[0], perna_esquerda.shape[1])) 

colagem[ 150: 150 + perna_direita_rot.shape[0], 150: 150 + perna_direita_rot.shape[1] ] = perna_direita_rot
# colagem[ 150: 150 + perna_esquerda_rot.shape[0], 150: 150 + perna_esquerda_rot.shape[1] ] = perna_esquerda_rot
# cv2.imshow('Tronco', tronco_rotacionado)
cv2.imshow('Colagem', colagem)
cv2.waitKey(0)
cv2.destroyAllWindows()