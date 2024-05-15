import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model

import constant

#def load_and_preprocess_images_input(image_path,start, end):
#    images = []
#    for index,file  in enumerate(os.listdir(image_path)):
#        if index < start:
#            continue
#        elif index >= end:
#            break
#        image = Image.open(image_path+"/"+file).convert("RGB")
#        images.append(image)
#    return np.array(images)
#
#def load_and_preprocess_images_output(image_path,start, end):
#    images = load_and_preprocess_images_input(image_path,start, end)
#    result = []
#    for image in images:
#        width, height, _ = image.shape
#        tmp = np.zeros((width, height), dtype=np.int32)
#        for x in range(width):
#            for y in range(height):
#                for label, value in constant.MAPPING_LABEL.items():
#                    if np.array_equal(image[x, y], label):
#                        tmp[x,y] = value
#                        break
#        result.append(tmp)
#
#    return np.array(result)

def load_and_preprocess_images_input(image_path, start, end):
    images = []
    filenames = sorted(os.listdir(image_path))
    for index, filename in enumerate(filenames[start:end]):
        image = Image.open(os.path.join(image_path, filename)).convert("RGB")
        images.append(np.array(image))
    return np.array(images)

def load_and_preprocess_images_output(image_path, start, end):
    images = load_and_preprocess_images_input(image_path, start, end)
    result = []
    for image in images:
        tmp = np.zeros(image.shape[:2], dtype=np.int32)
        for label, value in constant.MAPPING_LABEL.items():
            mask = np.all(image == np.array(label), axis=-1)
            tmp[mask] = value
        result.append(tmp)
    return np.array(result)

def run_model():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    image_height = 960
    image_width = 640
    num_channels = 3
    num_classes = 3
    num_epochs = 100

# Créer un modèle séquentiel
    model = models.Sequential()
    #model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Conv2D(4, (3, 3), activation='relu', padding='same',  input_shape=(image_height,image_width,num_channels)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2),strides=(1, 1),padding='same'))
    
    model.add(layers.Dense(16))
    model.add(layers.PReLU())
    model.add(layers.Conv2D(8, (2,2), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(1, 1),padding='same'))

    model.add(layers.Dense(16))
    model.add(layers.PReLU())
    model.add(layers.Conv2D(16, (2,2), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(1, 1),padding='same'))
    model.add(layers.Dense(4, activation='softmax'))
    model.add(layers.PReLU())


    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    train_images = load_and_preprocess_images_input(constant.TRAIN,0,50)
    train_labels = load_and_preprocess_images_output(constant.TRAIN_PROCESS_GROUND_TRUTH,0,50) 
    validation_images = load_and_preprocess_images_input(constant.VALIDATION,0,10)
    validation_labels = load_and_preprocess_images_output(constant.VALIDATION_PROCESS_GROUND_TRUTH,0,10) 

    
    model.summary()
    with tf.device('/device:GPU:0'):
        model.fit(train_images,train_labels, epochs=num_epochs, validation_data=(validation_images,validation_labels))
    test_loss, test_acc = model.evaluate(validation_images,validation_labels)
    return model,test_loss, test_acc

def save_my_model(model):
    save_model(model, 'model_test_2.keras')

def load_my_model(model):
    return load_model(model)



# 4. Validation du modèle


