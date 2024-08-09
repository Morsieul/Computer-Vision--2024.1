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



# Carregar a imagem
img = io.imread('/home/moe/Documents/images/balls.jpg')

# Aplicar o seam carving
img_v = seam_carving_v(img, 40)
img_h = seam_carving_h(img, 40)  # Reduz 20 costuras verticais
img_h_g = seam_carving_h_gambiarra(img, 40)

# cv2.imshow('original_image', img)
# cv2.imshow('new_image', new_image)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Mostrar a imagem original e a modificada
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_v)
ax[1].set_title('Vertical Seam Carved Image')
ax[1].axis('off')

ax[2].imshow(img_h)
ax[2].set_title('Horizontal Seam Carved Image')
ax[2].axis('off')

ax[3].imshow(img_h_g)
ax[3].set_title('Horizontal (  Rotation) Seam Carved Image')
ax[3].axis('off')

plt.tight_layout()
plt.show()
