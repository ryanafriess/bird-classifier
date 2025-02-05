import PIL
from PIL import Image
import pathlib
import tensorflow as tf

data_dir = pathlib.Path('bird_data/images/')  # Replace with the actual path

def main():
    print(data_dir)
    print(data_dir.exists())
    image_count = len(list(data_dir.glob('**/*.jpg')))
    print("Image count:", image_count)

    path = data_dir
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    class_names = train_ds.class_names
    print(class_names)
    
if __name__ == '__main__': main()