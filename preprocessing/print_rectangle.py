from PIL import Image, ImageDraw
import pandas as pd


image = Image.open('my_image.jpg')
df = pd.read_csv('coordonnees_rectangles.csv')
draw = ImageDraw.Draw(image)

# Dessiner les rectangles
for index, row in df.iterrows():
    x1, y1, x2, y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
    draw.rectangle([x1, y1, x2, y2], outline="white", width=3)
    print("j'ai fait un rectangle")


image = image.convert("RGB")
image.save('image_with_rectangles.jpg')
