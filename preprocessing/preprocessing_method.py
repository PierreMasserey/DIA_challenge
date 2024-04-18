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
