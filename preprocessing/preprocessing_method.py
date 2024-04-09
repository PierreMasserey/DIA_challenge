from PIL import Image
import numpy as np
import pandas as pd
from skimage import measure

image = Image.open('my_image.jpg')
image_np = np.array(image)

#trouver les contours
contours = measure.find_contours(np.array(image_np), 0.5)

rectangles = []

#add coordinate to the list
for contour in contours:
    y, x = contour[:, 0], contour[:, 1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    rectangles.append((x_min, y_min, x_max, y_max))


df = pd.DataFrame(rectangles, columns=['X1', 'Y1', 'X2', 'Y2'])
df.to_csv('coordonnees_rectangles.csv', index=False)
