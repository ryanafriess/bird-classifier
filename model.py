import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# dataset = ImageFolder(root="images/", transform=transforms)
dataset = ImageFolder(root="bird_data/images/", transform=transforms) 

dataset_length = len(dataset)
train_size = int(0.8 * dataset_length)
val_size = dataset_length - train_size

torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# Create DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create DataLoader for training and validation
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

images, labels = next(iter(dataloader))

# print(batch)
# for images, labels in dataloader:
#     print(images.shape)  # Output: torch.Size([32, 3, 128, 128])
#     print(labels)        # Output: tensor([0, 1, 1, 0, ...])
#     break  # Just checking one batch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
        self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 48, kernel_size = 3, padding = 1) #ask about naming

        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(48)

        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        # self.fc1 = torch.nn.Linear(48 * 28 * 28 * 28, 512)
        self.fc1 = torch.nn.Linear(48 * 56 * 56, 512)
        
        self.fc2 = torch.nn.Linear(512, 200)

    def forward(self, x):
        # Block 1
        x = self.conv1(x) #mention x fix
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Fully Connected Layers
        # print(x.shape)
        x = x.view(x.size(0), -1) #flatten (ill figure out why later)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        

        return x

def train_model(model = None, epochs = 5, save = True): #returns a model and saves model if save = True
    if model is None: model = CNN()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001) #Adam
    loss_func = nn.CrossEntropyLoss() #CrossEntropyLoss for multi-class classification (but only one class per image)

    print("Training started...")
    start_time = time.time()
    placeholder_time = time.time()
    for n in range(epochs):
        run_loss = 0.0

        for images, labels in trainloader: 
            #forward prop
            outputs = model(images)
            loss = loss_func(outputs, labels) #define loss_func ()

            optimizer.zero_grad() #define optimizer
            loss.backward()

            optimizer.step()
            run_loss += loss.item()

        print("Epoch #" + str(n+1) + ", Loss: " + str(run_loss /len(trainloader))) #average loss
        epoch_time = time.time() - placeholder_time
        print("Epoch completed after ", epoch_time, "seconds")
        placeholder_time = time.time()
        
    print("Training Over. Validation beginning...")
    total_train_time = time.time() - start_time
    print("Training over after ", total_train_time, "seconds")
    if(save): torch.save(model.state_dict(), 'bird_model_weights.pth')
    return model


def load_model(): #loads model from 'bird_model_weights.pth' and returns
    model = CNN()
    model.load_state_dict(torch.load('bird_model_weights.pth'))
    return model

def validate_model(model, valloader): #runs validation on model given valloader
    model.eval()
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in valloader: 
            outputs = model(images)
            _ , max_output = torch.max(outputs, 1)
            total_count += labels.size(0)
            correct_count += sum(1 for i in range(len(max_output)) if max_output[i] == labels[i])

    accuracy = 100 * correct_count / total_count
    print("Test Accuracy: " + str(accuracy))

def main():
    model = load_model()
    model = train_model(model=model, epochs=5, save=True)
    validate_model(model, valloader)
    
if __name__ == '__main__': main()