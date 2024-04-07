import numpy as np

def convert_binary_using_index(image,index):
    array = np.array(image)
    return np.where(array == index, 1, 0)

def compute_precision(image_1, image_2):
    width, height = image_1.size
    same_pixel = 0
    for i in range(width):
        for j in range(height):
            if image_1.getpixel((i,j)) == image_2.getpixel((i,j)) or (image_1.getpixel((i,j)) == 6 and image_2.getpixel((i,j)) not in LAYOUTS):
                same_pixel +=1
    return (same_pixel / (height * width)) *100

def compute_recall(image_1,image_2,index):
    new_image_1 = convert_binary_using_index(image_1,index)
    new_image_2 = convert_binary_using_index(image_2,index)
    number_positive_truth = np.sum(new_image_1)
    number_positive_classfied = np.sum(new_image_2)
    correct = np.sum(np.logical_and(new_image_1,new_image_2))
    false_negative = number_positive_truth - number_positive_classfied
    return correct / (correct + false_negative)

def compute_iou(image_1, image_2, index): ##Intersection over union by class
    new_image_1 = convert_binary_using_index(image_1,index)
    new_image_2 = convert_binary_using_index(image_2,index)
    intersection = np.logical_and(new_image_1,new_image_2)
    union = np.logical_or(new_image_1,new_image_2)
    return np.sum(intersection)/np.sum(union)