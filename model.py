import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

import constant

def load_and_preprocess_images_input(image_path,start, end):
    images = []
    for index,file  in enumerate(os.listdir(image_path)):
        if index < start:
            continue
        elif index >= end:
            break
        image = Image.open(image_path+"/"+file).convert("RGB")
        images.append(image)
    return np.array(images)

def load_and_preprocess_images_output(image_path,start, end):
    images = load_and_preprocess_images_input(image_path,start, end)
   
    
    result = []
    
    for image in images:
        width, height, _ = image.shape
        tmp = np.zeros((width, height), dtype=np.int32)
        for x in range(width):
            for y in range(height):
                for label, value in constant.MAPPING_LABEL.items():
                    if np.array_equal(image[x, y], label):
                        tmp[x,y] = value
                        break
        result.append(tmp)

    return np.array(result)

def run_model():

    image_height = 960
    image_width = 640
    num_channels = 3
    num_classes = 3
    num_epochs = 50
    train_images = load_and_preprocess_images_input(constant.ORG,0,50)
    train_labels = load_and_preprocess_images_output(constant.PROCESS_GROUND_TRUTH,0,50) 
    test_images = load_and_preprocess_images_input(constant.ORG,70,72)
    test_labels = load_and_preprocess_images_output(constant.PROCESS_GROUND_TRUTH,70,72) 


# Créer un modèle séquentiel
    model = models.Sequential()
    model.add(layers.Dense(16, input_shape=(image_height,image_width,num_channels)))
    model.add(layers.Dense(32)),
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu')),
    model.add(layers.Dense(32)),
    model.add(layers.Conv2D(16, (2, 2), padding='same', activation='relu')),
    model.add(layers.Dense(4))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  
    model.summary()
    model.fit(train_images,train_labels, epochs=num_epochs, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return model,test_loss, test_acc

def save_my_model(model):
    save_model(model, 'model_test.keras')

def load_my_model(model):
    return load_model(model)



# 4. Validation du modèle


