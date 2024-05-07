import os
import numpy as np
import random
from PIL import Image, ImageFilter
import measures
import model
import constant
import compute_iou
import image_to_csv


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
                else:    
                    rgb_value = np.array([i for i in  constant.MAPPING_LABEL if constant.MAPPING_LABEL[i]==(label)])
                rgb_image[x][y] = rgb_value

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
    
    return Image.fromarray(array_decoration), Image.fromarray(array_text_area), Image.fromarray(array_text_line)


def pre_process_all_ground_truth_images():
    pre_process_train()
    pre_process_validation()
    pre_process_test()

if __name__ == "__main__":
   #pre_process_all_ground_truth_images() ### A commenter si l'on a pas besoin de pre-process les ground truth: On met juste en noir tous les pixels dans les gorund truth que l'on a pas beson de détecter. Cela permet de ne pas entrainter notre model sur des points non nécessaire
   #trained_model, test_loss, test_acc = model.run_model()
   #print(test_loss, test_acc)
   #model.save_my_model(trained_model)
   my_trained_model = model.load_model('model_test_2.keras')
   input = model.load_and_preprocess_images_input(constant.TEST,0,1)
   prediction = my_trained_model.predict(input)
   predicted_rgb_image = post_process_prediction(prediction)
   image = Image.fromarray(np.uint8(predicted_rgb_image))
   image_decoration, image_text_area, image_text_line = separateColor(image)
   image_to_csv.image_to_csv(image, "tempname")
   print(os.path.realpath("./csv_groundtruth/utp-0110-061v.gif.csv"))
   #compute_iou.compute_iou("csv_groundtruth/utp-0110-061v.gif.csv","tempname.csv" )
   image_decoration.save("decoration.jpg")
   image_text_area.save("text_area.jpg")
   image_text_line.save("text_line.jpg")
   image.save("test.jpg")

