import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage import filters
from skimage import color

def calculate_energy(image):
    #Converte a imagem para tons de cinza
    gray_image = color.rgb2gray(image)

    #Calculando o gradiente das imagens 
    energy = np.abs(filters.sobel_h(gray_image))
    return energy
    
#Corte na Vertical

def find_seam_v(energy):
    #Energia acumulada ao longo do caminho minimo
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    # Preenchendo a matriz de energia acumulada
    for i in range(1, r):
        for j in range(c):
            # Bordas são tratadas separadamente
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            M[i, j] += min_energy

    return M, backtrack

def remove_seam_v(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(backtrack[-1])
    for i in reversed(range(r)):
        output[i, :, 0] = np.delete(image[i, :, 0], [j])
        output[i, :, 1] = np.delete(image[i, :, 1], [j])
        output[i, :, 2] = np.delete(image[i, :, 2], [j])
        j = backtrack[i, j]
    return output


#Corte na Horizontal

def find_seam_h(energy):
    # Energia acumulada ao longo do caminho mínimo
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    # Preenchendo a matriz de energia acumulada
    for j in range(1, c):  # Alterado para percorrer colunas
        for i in range(r):
            # Bordas são tratadas separadamente
            if i == 0:
                idx = np.argmin(M[i:i+2, j-1])
                backtrack[i, j] = idx + i
                min_energy = M[idx + i, j-1]
            else:
                idx = np.argmin(M[i-1:i+2, j-1])
                backtrack[i, j] = idx + i - 1
                min_energy = M[idx + i - 1, j-1]
            M[i, j] += min_energy

    return M, backtrack


def remove_seam_h(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r - 1, c, 3), dtype=image.dtype)
    i = np.argmin(backtrack[:, -1])
    for j in reversed(range(c)):
        output[:, j, 0] = np.delete(image[:, j, 0], [i])
        output[:, j, 1] = np.delete(image[:, j, 1], [i])
        output[:, j, 2] = np.delete(image[:, j, 2], [i])
        i = backtrack[i, j]
    return output



def seam_carving_v(image, num_seams):
    for _ in range(num_seams):
        energy = calculate_energy(image)
        M, backtrack = find_seam_v(energy)
        image = remove_seam_v(image, backtrack)
    return image


def seam_carving_h(image, num_seams):
    for _ in range(num_seams):
        energy = calculate_energy(image)
        M, backtrack = find_seam_h(energy)
        image = remove_seam_h(image, backtrack)
    return image

def seam_carving_h_gambiarra(image, num_seams):
    img = np.rot90(image, 1, (0, 1))
    img = seam_carving_v(img, num_seams)
    img = np.rot90(img, 3, (0, 1))
    return img


def select_roi(event, x, y, flags, param):
    global roi_pt1, roi_pt2, drawing, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_pt1 = (x, y)
        roi_pt2 = (x, y)
        roi_selected = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_pt2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_pt2 = (x, y)
        roi_selected = True

# Inicializar variáveis
roi_pt1 = (0, 0)
roi_pt2 = (0, 0)
drawing = False
roi_selected = False

# Carregar a imagem
img = io.imread('/home/moe/Documents/images/balls.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
img_copy = img.copy()

# Criar janela para exibição da imagem
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)

while True:
    img_display = img_copy.copy()
    if drawing:
        cv2.rectangle(img_display, roi_pt1, roi_pt2, (0, 255, 0), 2)

    cv2.imshow("Select ROI", img_display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif roi_selected:
        # Criar uma máscara para a área selecionada
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, roi_pt1, roi_pt2, 255, -1)

        # Aplicar seam carving apenas na área selecionada
        energy = calculate_energy(img)
        masked_energy = energy * (mask > 0)
        M, backtrack = find_seam_v(masked_energy)
        img = remove_seam_v(img, backtrack)

        # Atualizar a imagem original
        img_copy = img.copy()
        roi_selected = False

cv2.destroyAllWindows()

# Mostrar a imagem final
cv2.imshow("Seam Carved Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
