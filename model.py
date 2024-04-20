import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import constant

def load_and_preprocess_images_input(image_path,start, end):
    images = []
    for index,file  in enumerate(os.listdir(image_path)):
        if index <= start:
            continue
        elif index >= end:
            break
        image = Image.open(image_path+"/"+file).convert("RGB")
        images.append(image)
    return np.array(images)

def load_and_preprocess_images_output(image_path,start, end):
    images = load_and_preprocess_images_input(image_path,start, end)
    pixels = []
    for image in images:
        width = image.shape[0]
        height = image.shape[1]
        tmp = []
        for x in range(width):
            for y in range(height):
                tmp.append(image[x][y]) 
        pixels.append(tmp)
    return pixels

def run_model():

    image_height = 960
    image_width = 640
    num_channels = 3
    num_classes = 4
    num_epochs = 5
    train_images = load_and_preprocess_images_input(constant.ORG,0,30)
    train_labels = load_and_preprocess_images_output(constant.PROCESS_GROUND_TRUTH,0,30) 
    test_images = load_and_preprocess_images_input(constant.ORG, 31,40)
    test_labels = load_and_preprocess_images_output(constant.PROCESS_GROUND_TRUTH,31,40) 


# Créer un modèle séquentiel
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(num_classes, (1, 1), activation='softmax')  # Couche de sortie pour la segmentation
    ])


    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  

    model.fit(train_images,train_labels, epochs=num_epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return model,test_loss, test_acc






# 4. Validation du modèle


