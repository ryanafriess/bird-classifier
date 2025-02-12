import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import bird
import numpy as np
import PIL
import matplotlib.pyplot as plt
from PIL import Image

model = load_model("bird_model.h5")

# model.summary()

data_dir = "bird_data/images/"
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123, image_size=(224, 224), batch_size=32
)
class_names = train_ds.class_names  



def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)  # Use this for consistent preprocessing
    return img_array


def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)  # Get model prediction
    # print(prediction)
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get class index
    predicted_class = class_names[predicted_label]  # Map index to class name
    return predicted_class


# def visualize_prediction(image_path):
#     predicted_class, prediction = predict_image(image_path)

#     # Load and display the image
#     img = Image.open(image_path)
#     plt.figure(figsize=(8, 6))
    
#     # Show the image
#     plt.subplot(1, 2, 1)
#     plt.imshow(img)
#     plt.axis("off")
#     plt.title(f"Predicted: {predicted_class}")

#     # Show the prediction confidence for all classes
#     plt.subplot(1, 2, 2)
#     plt.barh(class_names, prediction, color="skyblue")
#     plt.xlabel("Confidence")
#     plt.title("Prediction Confidence")
#     plt.tight_layout()
#     plt.show()


def main():
    image_path = "humming2.jpg"
    predicted_class = predict_image(image_path)
    print("Predicted class:", predicted_class)
    
    # for images, labels in train_ds.take(1):  # Take one batch
    #     for i in range(32):
    #         if i == 0:
    #             image = images[i]
    #             image = tf.expand_dims(image, axis=0)
    #             label = labels[i]
    #             prediction = model.predict(image)
    #             predicted_label = np.argmax(prediction, axis=1)[0]  # Get class index
    #             predicted_class = class_names[predicted_label]  # Map index to class name
                
    #             print("Predicted class:", predicted_class)
    #             print("Correct class:", label)
    
if __name__ == '__main__': main()


