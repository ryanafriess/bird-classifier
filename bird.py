import PIL
from PIL import Image
import pathlib

data_dir = pathlib.Path('nabirds/images/')  # Replace with the actual path

print(data_dir)
print(data_dir.exists())


image_count = len(list(data_dir.glob('**/*.jpg')))
print("Image count:", image_count)

house_finch = list(data_dir.glob('0997/*'))
img = PIL.Image.open(str(house_finch[0]))
img.show()