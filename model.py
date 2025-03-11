import PIL
from PIL import Image
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np


data_dir = pathlib.Path('images/')

def load_img_data(path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = 'training',
        seed = 123,
        image_size = (224,224),
        batch_size = 32
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = 'validation',
        seed = 123,
        image_size = (224,224),
        batch_size = 32
    )
    
    return train_ds, val_ds

def feature_scale(train_ds):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomCrop(180, 180, seed = None),
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.RandomContrast(factor = 0.5, seed = None),
        tf.keras.layers.RandomFlip('horizontal'),
        # tf.keras.layers.RandomTranslation(height_factor = 0.1, width_factor = 0.1),
        # tf.keras.layers.RandomRotation(0.2),
    ])

    normalization = tf.keras.layers.Rescaling(1./255, input_shape = (224, 224, 3))

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
 
    return train_ds

def feature_scale_val_ds(val_ds):
    normalization = tf.keras.layers.Rescaling(1./255, input_shape = (224, 224, 3))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))

    return val_ds


def show_images(train_ds, class_names):
    image_batch, label_batch = next(iter(train_ds))
    image_batch = image_batch.numpy()

    plt.figure(figsize=(10, 10))
   
    for i in range(9):  # Show 9 images
        plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i])  # Display image
        plt.title(class_names[label_batch[i].numpy()]) 
        plt.axis("off")  # Hide axes
    
    plt.show()



def train_model(model, train_ds, val_ds, epoch_num):

    learning_rate = 0.0001

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = ['accuracy']
    )

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)


    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = epoch_num,
        callbacks = [early_stopping]
    )

    model.save('bird_model.keras')

def train_from_file(filename, train_ds, val_ds, epoch_num):
    model = load_model(filename)
    train_model(model, train_ds, val_ds, epoch_num)
    model.save('bird_model.keras')



def main():
    train_ds, val_ds = load_img_data(data_dir)
    class_names = train_ds.class_names
    train_ds = feature_scale(train_ds)


    # model = tf.keras.Sequential([
    # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    # layers.MaxPooling2D(2, 2),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D(2, 2),
    # layers.Conv2D(128, (3, 3), activation='relu'),
    # layers.MaxPooling2D(2, 2),
    # layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dense(len(class_names), activation='softmax')  # Output layer  # Output layer
    # ])

    base_model = tf.keras.applications.MobileNetV2(input_shape = (224, 224, 3),
                                                   include_top = False,
                                                   weights = 'imagenet')
    base_model.trainable = True

    for layer in base_model.layers[:-10]:
        layer.Trainable = False
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(class_names), activation='softmax')


    inputs = tf.keras.Input(shape = (224, 224, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training = False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    tf.keras.layers.Dense(256, activation = 'relu')(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)


    #show_images(train_ds, class_names)
    #train_model(model, train_ds, val_ds, 10)
    train_from_file('bird_model.keras', train_ds, val_ds, 1)
    

   



if __name__ == '__main__': main()