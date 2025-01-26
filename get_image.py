import PIL
from PIL import Image
import pathlib

data_dir = pathlib.Path('nabirds/images/')  # Replace with the actual path

print(data_dir)
print(data_dir.exists())


image_count = len(list(data_dir.glob('**/*.jpg')))
print("Image count:", image_count)


def clean_num(num): #changes number from int to '{num}/*' form
    if num < 0 or num >= 10000:
        return "invalid number"
    return f"{num:04d}/*"

def show_image(num):
    num_str = clean_num(num)
    print(num_str)

    sample_image = list(data_dir.glob(num_str))
    img = PIL.Image.open(str(sample_image[0]))
    img.show()
    
def main():
    while True:
        num = int(input("What number bird would you like to see? (-1 to quit): "))
        if num == -1 or num > 1010 or num < 295:
            break
        show_image(num)
        
if __name__ == '__main__': main()