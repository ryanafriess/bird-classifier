import PIL
import pathlib

data_dir = pathlib.Path('nabirds/images/')  # Replace with the actual path

print(data_dir)
print(data_dir.exists())


image_count = len(list(data_dir.glob('**/*.jpg')))
print("Image count:", image_count)

# roses = list(data_dir.glob('/*'))
# PIL.Image.open(str(roses[0]))