import os
import numpy as np
import random
from PIL import Image, ImageFilter
import measures
import model
import constant
import compute_iou
import image_to_csv
import cv2


erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'


def random_classifier(image):
    width, height = image.size
    classified_image = Image.new("P",(width, height))
    for i in range(width):
        for j in range(height):
            classified_image.putpixel((i,j),random.choice(constant.LAYOUTS))
    return classified_image

def pre_process_ground_truth(image):
    image_rgb = image.convert("RGB")
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b = image_rgb.getpixel((x, y))
            pixel_color = (r, g, b)
            if pixel_color not in constant.LAYOUTS:
                image.putpixel((x, y), (0,0,0))  # Mettre à noir le pixel
    return image

def post_process_prediction(predictions):
    _,height, width, classes = predictions.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    # Parcourir chaque pixel de l'image étiquetée
    for prediction in predictions:
        for x in range(height):
            for y in range(width):
                label = np.argmax(prediction[x, y])
                if label == 0:
                    rgb_value = (0,0,0)
                elif label == 1:
                    rgb_value = constant.DECORATION
                elif label == 2:
                    rgb_value = constant.TEXT_AREA
                else:
                    rgb_value = constant.TEXT_LINE
                
                rgb_image[x][y] = np.array(rgb_value)

        return rgb_image
    
def pre_process_train():
    for file in os.listdir(constant.TRAIN_GROUND_TRUTH):
        try:
            ground_truth_image = Image.open(constant.TRAIN_GROUND_TRUTH+"/"+file)
            pre_process_ground_truth(ground_truth_image).save(constant.TRAIN_PROCESS_GROUND_TRUTH+"/"+file.replace(".jpg",".gif"))
        except:
            continue

def pre_process_validation():
    for file in os.listdir(constant.VALIDATION_GROUND_TRUTH):
        try:
            ground_truth_image = Image.open(constant.VALIDATION_GROUND_TRUTH+"/"+file)
            pre_process_ground_truth(ground_truth_image).save(constant.VALIDATION_PROCESS_GROUND_TRUTH+"/"+file.replace(".jpg",".gif"))
        except:
            continue

def pre_process_test():
    for file in os.listdir(constant.TEST_GROUND_TRUTH):
        try:
            ground_truth_image = Image.open(constant.TEST_GROUND_TRUTH+"/"+file)
            pre_process_ground_truth(ground_truth_image).save(constant.TEST_PROCESS_GROUND_TRUTH+"/"+file.replace(".jpg",".gif"))
        except:
            continue  
def separateColor(image):
    np_image = np.array(image)
    array_decoration = np.copy(np_image)
    array_text_area =  np.copy(np_image)
    array_text_line =  np.copy(np_image)

    decoration = np.array(constant.DECORATION)
    text_area = np.array(constant.TEXT_AREA)
    text_line = np.array(constant.TEXT_LINE)

    mask_decoration = np.all(array_decoration == decoration,axis=-1)
    mask_text_area = np.all(array_text_area == text_area ,axis=-1)
    mask_text_line = np.all(array_text_line == text_line ,axis=-1)

    array_decoration[np.where(~mask_decoration)]=[0,0,0]
    array_text_area [np.where(~mask_text_area) ] =[0,0,0]
    array_text_line [np.where(~mask_text_line) ] = [0,0,0]
    
    return array_decoration, array_text_area, array_text_line


def pre_process_all_ground_truth_images():
    pre_process_train()
    pre_process_validation()
    pre_process_test()


def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE
    
def erosion(image):
    erosion_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = morph_shape(cv2.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))

    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))

    return  cv2.erode(image, element)


def dilatation(image):
    dilatation_size = cv2.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = morph_shape(cv2.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))
    element = cv2.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),(dilatation_size, dilatation_size))
    return cv2.dilate(image, element)



if __name__ == "__main__":
   #pre_process_all_ground_truth_images() ### A commenter si l'on a pas besoin de pre-process les ground truth: On met juste en noir tous les pixels dans les gorund truth que l'on a pas beson de détecter. Cela permet de ne pas entrainter notre model sur des points non nécessaire
   #trained_model, test_loss, test_acc = model.run_model()
   ##print(test_loss, test_acc)
   #model.save_my_model(trained_model)
   my_trained_model = model.load_model('model_test_2.keras')
   Ious = []
   percentages = []
   for filename in os.listdir(constant.TEST):
       images = []
       image = Image.open(os.path.join(constant.TEST, filename)).convert("RGB")
       images.append(np.array(image)) 
       prediction = my_trained_model.predict(np.array(images))

       
       predicted_rgb_image = post_process_prediction(prediction)
       
       image = Image.fromarray(np.uint8(predicted_rgb_image),'RGB')
       ##Color separation
       array_decoration, array_text_area, array_text_line = separateColor(image)

       image_decoration = array_decoration[:, :, ::-1].copy()
       image_text_area = array_text_area[:, :, ::-1].copy()
       image_text_line = array_text_line[:, :, ::-1].copy()
       
       #Remove noise for each image  
       binary_image_decoration = cv2.cvtColor(image_decoration, cv2.COLOR_BGR2GRAY)
       _, binary_image_decoration = cv2.threshold(binary_image_decoration, 0, 255, cv2.THRESH_BINARY)
       
       binary_text_area = cv2.cvtColor(image_text_area, cv2.COLOR_BGR2GRAY)
       _, image_binaire_text_area = cv2.threshold(binary_text_area, 127, 255, cv2.THRESH_BINARY)
       
       binary_text_line = cv2.cvtColor(image_text_line, cv2.COLOR_BGR2GRAY)
       _, image_binaire_text_line = cv2.threshold(binary_text_line, 0, 255, cv2.THRESH_BINARY)
       
       kernel_open = np.ones((2,2), dtype=np.uint8)
       kernel_medium = np.ones((3,3), dtype=np.uint8)
       kernel_close = np.ones((4,4), dtype=np.uint8)
       
       denoised_decoration = cv2.morphologyEx(binary_image_decoration, cv2.MORPH_OPEN, kernel_close)
       denoised_text_area =  cv2.morphologyEx(image_binaire_text_area, cv2.MORPH_OPEN, kernel_close)
       denoised_text_line =  cv2.morphologyEx(image_binaire_text_line, cv2.MORPH_OPEN, kernel_open)
       
       denoised_decoration = cv2.morphologyEx(denoised_decoration, cv2.MORPH_CLOSE, kernel_medium)
       denoised_text_area =  cv2.morphologyEx(denoised_text_area, cv2.MORPH_CLOSE, kernel_medium)
       denoised_text_line =  cv2.morphologyEx(denoised_text_line, cv2.MORPH_CLOSE, kernel_close)


       #
       ##Put image in corresponding color 
       decoration = np.zeros((array_decoration.shape), dtype=np.uint8)
       text_area = np.zeros((array_text_area.shape), dtype=np.uint8)
       text_line = np.zeros((array_text_line.shape),dtype=np.uint8)
       #
       
       decoration[denoised_decoration == 255] = np.array(constant.DECORATION) 
       text_area[denoised_text_area == 255] = np.array(constant.TEXT_AREA)
       text_line[denoised_text_line == 255] = np.array(constant.TEXT_LINE)
       
       
       #
       height, width, classes = np.uint8(image).shape
       final = np.zeros((height, width, classes),dtype=np.uint8)
       mask_decoration = np.all(decoration == constant.DECORATION,axis=-1)
       mask_text_area = np.all(text_area == constant.TEXT_AREA ,axis=-1)
       mask_text_line = np.all(text_line == constant.TEXT_LINE ,axis=-1)

       final[np.where(mask_decoration)]=constant.DECORATION
       final[np.where(mask_text_area)]=constant.TEXT_AREA
       final[np.where(mask_text_line)]=constant.TEXT_LINE
       

       image_final = Image.fromarray(final,'RGB')
       image_to_csv.image_to_csv(image_final, "tempname")
       image_to_csv.image_to_csv(Image.open(constant.TEST_GROUND_TRUTH+"/"+filename.replace(".jpg",".gif")),"tempname2")
       iou = compute_iou.compute_iou("tempname2.csv","tempname.csv" )
       Ious.append(iou)
    
       test =  np.array(Image.open(constant.TEST_GROUND_TRUTH+"/"+filename.replace(".jpg",".gif")).convert("RGB"), dtype=np.uint8)
       
       mask1 = np.all(final == [0, 0, 0], axis=-1)  # Masque pour les pixels noirs dans image1
       mask2 = np.all(test == [0, 0, 0], axis=-1)


       black_pixels_mask = np.logical_or(mask1, mask2)

       # Inverser le masque pour obtenir un masque pour les pixels non-noirs
       non_black_pixels_mask = np.invert(black_pixels_mask)

       # Compter les pixels identiques en excluant les pixels noirs

       total_pixels_identiques = np.sum(np.all(final[non_black_pixels_mask] == test[non_black_pixels_mask], axis=-1))
       pourcentage_pixels_identiques =( total_pixels_identiques / np.count_nonzero(final[non_black_pixels_mask])) * 100
       percentages.append(pourcentage_pixels_identiques)
       print("Nombre de pixels identiques :", total_pixels_identiques)
       print("Pourcentage de pixels identiques :", pourcentage_pixels_identiques)
       stop = 1
   print("Mean of Ious :", np.mean(Ious)) 
   print("Mean of percentages of corresponding pixels :", np.mean(percentages))  
       
  



   
   
   

