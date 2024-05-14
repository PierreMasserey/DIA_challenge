from PIL import Image, ImageDraw
import cv2
import numpy as np
import csv
import os

dossier_images = "ground_truth"

# Liste globale pour stocker toutes les coordonnées des coins
all_corners_with_color = []

def image_to_csv(image, nom_fichier):
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

    def add_corners_with_color_to_list(corners, color_name, index=None):
        for corner in corners:
            x, y = corner
            if index is not None:
                color_name += str(index)
            all_corners_with_color.append((x, y, color_name))
        


    # Définir une fonction pour enregistrer toutes les coordonnées avec leur couleur dans un fichier CSV
    def save_corners_with_color_to_csv(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Y', 'Color'])  # Écrire l'en-tête du fichier CSV
            for corner in all_corners_with_color:
                writer.writerow(corner)
        all_corners_with_color.clear()

    def process_corners(image, color_name):
        # Convertir l'image en niveaux de gris
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Binariser l'image pour obtenir un masque
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # Trouver les contours dans le masque
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Créer un objet ImageDraw pour dessiner sur l'image
        draw = ImageDraw.Draw(image)

        # Dessiner les coins détectés et ajouter les coordonnées à la liste
        for i, contour in enumerate(contours):
            for point in contour:
                x, y = point[0]
                add_corners_with_color_to_list([(x, y)], color_name, index=i)  # Ajouter les coordonnées avec index
                draw.ellipse((x-2, y-2, x+2, y+2), fill="red")  # dessiner un petit cercle aux coordonnées du coin
        
        # Ajouter une ligne vide pour séparer les zones
        all_corners_with_color.append(("", "", ""))

        return image

    # Appeler la fonction process_corners pour chaque image
    process_corners(red_image, "red")
    process_corners(blue_image, "blue")
    process_corners(light_blue_image, "lightblue")
    process_corners(purple_image, "purple")

    # Enregistrer toutes les coordonnées avec leur couleur dans un fichier CSV
    save_corners_with_color_to_csv(nom_fichier  +".csv")


tempimage = Image.open("utp-0110-023v.gif")
image_to_csv(tempimage, "pourvoir")