import os
import numpy as np
import random
from PIL import Image
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
                image.putpixel((x, y), (255,192,203))  # Mettre à noir le pixel
    return image

def post_process_prediction(predictions):
    _,height, width, classes = predictions.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Parcourir chaque pixel de l'image étiquetée
    for prediction in predictions:
        for x in range(height):
            for y in range(width):
                label = np.argmax(prediction[x, y])
                if label == 4:
                    rgb_value = (255,255,255)
                else:    
                    rgb_value = np.array([i for i in  constant.MAPPING_LABEL if constant.MAPPING_LABEL[i]==label])
                rgb_image[x][y] = rgb_value

        return rgb_image

def pre_process_all_ground_truth_images():
     for file in os.listdir(constant.ORG):
        ground_truth_file = constant.GROUND_TRUTH + file.replace(".jpg",".gif")
        ground_truth_image = Image.open(ground_truth_file)
        pre_process_ground_truth(ground_truth_image).save(constant.PROCESS_GROUND_TRUTH+"/"+file.replace(".jpg",".gif"))

if __name__ == "__main__":
   #pre_process_all_ground_truth_images() ### A commenter si l'on a pas besoin de pre-process les ground truth: On met juste en noir tous les pixels dans les gorund truth que l'on a pas beson de détecter. Cela permet de ne pas entrainter notre model sur des points non nécessaire
   #trained_model, test_loss, test_acc = model.run_model()
   #print(test_loss, test_acc)
   #model.save_my_model(trained_model)
   my_trained_model = model.load_model('model_test.keras')
   input = model.load_and_preprocess_images_input(constant.ORG,80,81)
   prediction = my_trained_model.predict(input)
   predicted_rgb_image = post_process_prediction(prediction)
   image = Image.fromarray(np.uint8(predicted_rgb_image))
   image_to_csv.image_to_csv(image, "tempname")
   compute_iou.compute_iou("csv_groundtruth/tempname.csv","tempname.csv" )

   image.save("test.jpg")

