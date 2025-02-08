import PIL
from PIL import Image
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

data_dir = pathlib.Path('bird_data/images/')  # Replace with the actual path
    
def get_class_names():
    path = data_dir
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32,
    )
    
    class_names = train_ds.class_names
    return class_names

def display_images(batch, class_names):
    plt.figure(figsize=(12, 12))
    grid_size = 8
    rows = 4

    images, labels = batch[0]
    for i in range(32):
        ax = plt.subplot(rows, grid_size, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]].split(".")[0])
        plt.axis("off")
    plt.show()
    
def train_model(train_ds):
    #creates the tensorflor CNN model
    model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')  # Output layer
    ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(train_ds, epochs=10)
    
    model.save("bird_model.h5")
    return model

def train_from_file(filename, epoch_num, train_ds):
    model = load_model(filename)
    model.compile(optimizer='adam',
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])
    model.fit(train_ds, epochs=epoch_num)
    model.save("bird_model.h5")
    return model

def main():
    # print(data_dir)
    # print(data_dir.exists())
    # image_count = len(list(data_dir.glob('**/*.jpg')))
    # print("Image count:", image_count)

    path = data_dir
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32,
    )
    
    class_names = train_ds.class_names
    print("Number of batches: ", len(train_ds))
    
    # for images, labels in train_ds.take(1):  # Take one batch
    #     # display_images([(images, labels)], class_names)
    #     print("Image Tensor Shape:", images.shape)
    #     print("Labels:", labels.numpy())
    #     for i in range(32):
    #         if i == 0:
    #             image = images[i].numpy().astype("uint8")
    #             rgb = image[112, 112]
    #             print("RGB value: ", rgb)
    
    model = train_from_file("bird_model.h5", 10, train_ds)
    # model = train_model(train_ds)
    # model = load_model("bird_model.h5")
    for images, labels in train_ds.take(1):
        predictions = model.predict(images)
        predicted_labels = tf.argmax(predictions, axis=1).numpy()
        print("Predicted labels:", predicted_labels)
        print("Actual labels:", labels.numpy())
        prediction = {}
        for i in range(32):
             prediction[int(predicted_labels[i])] = int(labels.numpy()[i])
        print(prediction)

    
if __name__ == '__main__': main()