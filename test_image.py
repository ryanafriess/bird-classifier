import tensorflow as tf
from tensorflow.keras.models import load_model
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
    img = Image.open(image_path)  # Open image
    img = img.resize((224, 224))  # Resize to match model input size
    img = np.array(img)

    return img

def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)  # Get model prediction
    predicted_label = np.argmax(prediction, axis=1)[0]  # Get class index
    predicted_class = class_names[predicted_label]  # Map index to class name
    return predicted_class

def main():
    image_path = "glaucous.jpg"
    predicted_class = predict_image(image_path)
    print("Predicted class:", predicted_class)
    
if __name__ == '__main__': main()


