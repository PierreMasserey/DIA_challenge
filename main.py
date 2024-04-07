import os
import numpy as np
import random
from PIL import Image
import measures
##Path constant
IMAGES = "./layout_analysis"
GROUND_TRUTH = IMAGES + "/ground_truth/"
ORG = IMAGES + "/images/"

## INDEX CORRESPONDANCE
TEXT_AREA = 0 # Yellow
LARGE_CAPTIAL = 1 # light blue
FILLER = 2 #Green
SMALL_CAPITAL = 3 #MAGENTA
DECORATION = 4 #RED
TEXT_LINE = 5 #BLUE
BACKGROUND = 6 #Black

LAYOUTS = [DECORATION, TEXT_AREA, TEXT_LINE]


def random_classifier(image):
    width, height = image.size
    classified_image = Image.new("P",(width, height))
    for i in range(width):
        for j in range(height):
            classified_image.putpixel((i,j),random.choice(LAYOUTS))
    return classified_image


if __name__ == "__main__":
    for file in os.listdir(ORG):
        ground_truth_file = GROUND_TRUTH + file.replace(".jpg",".gif")
        if not os.path.exists(ground_truth_file):
            raise ValueError("Ground truth file not found")
        classified_image = random_classifier(Image.open(ORG+"/"+file))
        ground_truth_image = Image.open(ground_truth_file)
        print(measures.compute_precision(classified_image,ground_truth_image))
        print(measures.compute_iou(classified_image,ground_truth_image,DECORATION))
        print(measures.compute_iou(classified_image,ground_truth_image,TEXT_AREA))
        print(measures.compute_iou(classified_image,ground_truth_image,TEXT_LINE))
        print(measures.compute_recall(classified_image,ground_truth_image,DECORATION))
        print(measures.compute_recall(classified_image,ground_truth_image,TEXT_AREA))
        print(measures.compute_recall(classified_image,ground_truth_image,TEXT_LINE))
        break