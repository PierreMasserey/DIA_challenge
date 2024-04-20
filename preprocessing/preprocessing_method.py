<<<<<<< HEAD
from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
import csv

# Liste globale pour stocker toutes les coordonnées des coins
all_corners_with_color = []

# Charger l'image
image = Image.open("my_image.gif")

# Convertir l'image en mode RGB si elle n'est pas déjà en RGB
image = image.convert("RGB")

# Obtenir les dimensions de l'image
largeur, hauteur = image.size

blue_image = Image.new('RGB', (largeur, hauteur), color='black')
red_image = Image.new('RGB', (largeur, hauteur), color='black')
light_blue_image = Image.new('RGB', (largeur, hauteur), color='black')
purple_image = Image.new('RGB', (largeur, hauteur), color='black')




# Parcourir tous les pixels de l'image
for x in range(largeur):
    for y in range(hauteur):
        # Récupérer la couleur du pixel
        r, g, b = image.getpixel((x, y))
        
        # Vérifier si le pixel est rouge pur (255, 0, 0)
        if (r, g, b) == (255, 0, 0):
            red_image.putpixel((x, y), (255, 0, 0))
        elif (r, g, b) == (0, 0, 255):
            blue_image.putpixel((x, y), (0, 0, 255))
        elif (r, g, b) == (0, 255, 255):
            light_blue_image.putpixel((x, y), (0, 255, 255))
        elif (r, g, b) == (255, 0, 255):
            purple_image.putpixel((x, y), (255, 0, 255))

            # Laisser la couleur rouge intacte
            
        else:
            pass
            # Convertir tout autre pixel en noir



def add_corners_with_color_to_list(corners, color_name):
    for corner in corners:
        x, y = corner[0]
        all_corners_with_color.append((x, y, color_name))

# Définir une fonction pour enregistrer toutes les coordonnées avec leur couleur dans un fichier CSV
def save_corners_with_color_to_csv(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y', 'Color'])  # Écrire l'en-tête du fichier CSV
        for corner in all_corners_with_color:
            writer.writerow(corner)  # Écrire les coordonnées de chaque coin avec leur couleur

def process_corners(image, color_name):
    # Convertir l'image en niveaux de gris
    gray_image = image.convert("L")

    # Détection de contours
    contour_image = gray_image.filter(ImageFilter.FIND_EDGES)

    # Détection de coins
    corners = cv2.goodFeaturesToTrack(np.array(contour_image), maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Ajouter les coordonnées des coins avec leur couleur à la liste globale
    add_corners_with_color_to_list(corners, color_name)

    # Créer un objet ImageDraw pour dessiner sur l'image
    draw = ImageDraw.Draw(image)

    # Dessiner les coins détectés
    for corner in corners:
        x, y = corner[0]
        draw.ellipse((x-2, y-2, x+2, y+2), fill="red")  # dessiner un petit cercle aux coordonnées du coin
    
    return image

# Appeler la fonction process_corners pour chaque image
process_corners(red_image, "red")
process_corners(blue_image, "blue")
process_corners(light_blue_image, "light_blue")
process_corners(purple_image, "purple")

# Enregistrer toutes les coordonnées avec leur couleur dans un fichier CSV
save_corners_with_color_to_csv("all_corners_with_color.csv")
=======
from PIL import ImageFilter, Image
from PIL import ImageDraw
import cv2
import numpy as np

# Charger l'image
image_path = "my_image.jpg"
image = Image.open(image_path)

# Convertir l'image en niveaux de gris
gray_image = image.convert("L")

# Détection de contours
contour_image = gray_image.filter(ImageFilter.FIND_EDGES)

# Détection de coins
corners = cv2.goodFeaturesToTrack(np.array(contour_image), maxCorners=100, qualityLevel=0.5, minDistance=50)

# Créer un objet ImageDraw pour dessiner sur l'image
draw = ImageDraw.Draw(image)

# Dessiner les coins détectés
for corner in corners:
    x, y = corner[0]
    draw.ellipse((x-2, y-2, x+2, y+2), fill="white")  # dessiner un petit cercle aux coordonnées du coin

# Enregistrer l'image avec les coins détectés
image.save("image_avec_corners.png")
>>>>>>> 96fd890ad0ab1e7fcf8c33a5352b858d5987cdd0