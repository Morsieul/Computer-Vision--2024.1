import cv2
import numpy as np
import matplotlib.pyplot as mp 


def detect_corners(image_path):
    
    img = cv2.imread(image_path)

    if img is None:
        print("Não foi possível abrir ou não foi encontrada\n")
        return 
    
    image = np.copy(img)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray_image)

    harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    # Mostrar imagem original e a imagem com cantos detectados

    mp.figure(figsize = (12, 6))
    mp.subplot(1, 2, 1)
    mp.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mp.title("Original")
    mp.axis('off')

    mp.subplot(1, 2, 2)
    mp.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mp.title('Cantos detectados')
    mp.axis('off')

    mp.show()


if __name__ == "__main__":
    detect_corners('./puppy.jpg')