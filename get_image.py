import PIL
from PIL import Image
import pathlib
import random
import classes_map


data_dir = pathlib.Path('nabirds/images/')  # Replace with the actual path 
bird_dict = classes_map.build_dict()

# print(data_dir)
# print(data_dir.exists())


image_count = len(list(data_dir.glob('**/*.jpg')))
print("Image count:", image_count)


def clean_num(num): #changes number from int to '{num}/*' form
    if num < 0 or num >= 10000:
        print("invalid number")
        return -1
    return f"{num:04d}/*"

def show_image(num): #shows a random image given a number
    num_str = clean_num(num)
    print("Showing random image of " + bird_dict[num])

    sample_image = list(data_dir.glob(num_str))
    img = PIL.Image.open(str(sample_image[random.randrange(len(sample_image))]))
    img.show()
    
def main():
    while True:
        num = int(input("What number bird would you like to see? (-1 to quit): "))
        if num == -1 or num > 1010 or num < 295:
            break
        show_image(num)
        
if __name__ == '__main__': main()