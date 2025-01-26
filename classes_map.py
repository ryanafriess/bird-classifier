#File meant to map class label (numbers) to species using classes.txt

def build_dict():
    classes = open("classes.txt").read().splitlines()
    # print(classes)

    bird_dict = {}

    for bird in classes:
        bird_split = bird.split()
        num = int(bird_split[0])
        species = " ".join(bird_split[1:])
        
        bird_dict[num] = species
        
    return bird_dict
    
def get_species(bird_dict, num): #assumes num is in bird_dict
    return bird_dict[num]
    

def main(): 
    bird_dict = build_dict()
    while True:
        num = int(input("Enter number for species (-1 to quit): "))
        if num == -1 or num > 1010:
            break
        print(bird_dict[num])
    
if __name__ == '__main__': main()
    
