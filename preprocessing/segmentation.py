from PIL import ImageFilter, Image
from PIL import ImageDraw
import cv2
import numpy as np

# Charger l'image
image = Image.open("my_image.jpg")

# Convertir l'image en mode RGB si elle n'est pas déjà en RGB
image = image.convert("RGB")

# Obtenir les dimensions de l'image
largeur, hauteur = image.size

image_red = Image.new('RGB', (largeur, hauteur), color='black')
image_light_blue = Image.new('RGB', (largeur, hauteur), color='black')
image_blue = Image.new('RGB', (largeur, hauteur), color='black')
image_purple = Image.new('RGB', (largeur, hauteur), color='black')

# Parcourir tous les pixels de l'image
for x in range(largeur):
    for y in range(hauteur):
        # Récupérer la couleur du pixel
        r, g, b = image.getpixel((x, y))
        
        # Vérifier si le pixel est rouge pur (255, 0, 0)
        if (r, g, b) == (255, 0, 0):
            image_red.putpixel((x, y), (255, 0, 0))
        elif (r, g, b) == (0, 255, 255):
            image_light_blue.putpixel((x, y), (0, 255, 255))
        elif (r, g, b) == (0, 0, 255):
            image_blue.putpixel((x, y), (0, 0, 255))
        elif (r, g, b) == (255, 0, 255):
            image_purple.putpixel((x, y), (255, 0, 255))
        else : pass    
            

def find_corner(image):
    gray_image = image.convert("L")

    # Détection de contours
    contour_image = gray_image.filter(ImageFilter.FIND_EDGES)

    # Détection de coins
    corners = cv2.goodFeaturesToTrack(np.array(contour_image), maxCorners=100, qualityLevel=0.2, minDistance=10)

    # Créer un objet ImageDraw pour dessiner sur l'image
    draw = ImageDraw.Draw(image)

    # Dessiner les coins détectés
    for corner in corners:
        x, y = corner[0]
        draw.ellipse((x-2, y-2, x+2, y+2), fill="white")  # dessiner un petit cercle aux coordonnées du coin

    return image



# Enregistrer l'image modifiée

find_corner(image_red).save("image_rouge.jpg")
find_corner(image_light_blue).save("image_bc.jpg")
find_corner(image_blue).save("image_bleue.jpg")
find_corner(image_purple).save("image_purple.jpg")

