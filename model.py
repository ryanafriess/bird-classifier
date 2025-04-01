import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from collections import Counter


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# dataset = ImageFolder(root="images/", transform=transforms)
dataset = ImageFolder(root="images/", transform=transforms) 
NUM_CLASSES = len(dataset.classes)
# print(len(dataset.classes))




dataset_length = len(dataset)
print("Dataset length:", dataset_length)
train_size = int(0.6 * dataset_length)
val_size = int(0.2 * dataset_length)
test_size = dataset_length - train_size - val_size

torch.manual_seed(50)
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
print("______BEGIN")
print(Counter(train_labels))
print("END_______")


# print("Validation set size:", len(val_dataset))
# print("Sample train labels:", [train_dataset[i][1] for i in range(5)])
# print("Sample val labels:", [val_dataset[i][1] for i in range(5)])

#SubsetRandomSampler

# Create DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create DataLoader for training and validation
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
print(Counter(val_labels))

# images, labels = next(iter(dataloader))

# print(batch)
# for images, labels in dataloader:
#     print(images.shape)  # Output: torch.Size([32, 3, 128, 128])
#     print(labels)        # Output: tensor([0, 1, 1, 0, ...])
#     break  # Just checking one batch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        #block 1
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        #block 2
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1) #ask about naming
        self.bn2 = torch.nn.BatchNorm2d(64)

        #block 3
        self.conv3 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        #block 4
        self.conv4 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.bn4 = torch.nn.BatchNorm2d(256)

        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        #Fully connected layers
        self.fc1 = torch.nn.Linear(256 * 14 * 14, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, NUM_CLASSES)

        #droput layer 
        self.dropout = torch.nn.Dropout(0.3)

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

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # Fully Connected Layers
        # print(x.shape)
        x = x.view(x.size(0), -1) #flatten (ill figure out why later)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        

        return x

def train_model(model = None, epochs = 5, save = True): #returns a model and saves model if save = True
    if model is None: model = CNN()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001) #Adam
    schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    loss_func = nn.CrossEntropyLoss() #CrossEntropyLoss for multi-class classification (but only one class per image)

    print("Training started...")
    start_time = time.time()
    placeholder_time = time.time()
    for n in range(epochs):
        model.train()
        run_loss = 0.0

        for images, labels in trainloader: 
            #forward prop
            outputs = model(images)
            loss = loss_func(outputs, labels) #define loss_func ()

            optimizer.zero_grad() #define optimizer
            loss.backward()

            optimizer.step()
            run_loss += loss.item()

        print("Epoch #" + str(n+1) + ", Loss: " + str(run_loss / len(trainloader))) #average loss
        epoch_time = time.time() - placeholder_time
        print("Epoch completed after ", epoch_time, "seconds")
        placeholder_time = time.time()
        
        ### epoch validation ###
        model.eval()
        pred_counts = []
        correct_count = 0
        total_count = 0
        with torch.no_grad():
            for images, labels in valloader:
                outputs = model(images)
                _, max_output = torch.max(outputs, 1)
                pred_counts.extend(max_output.tolist())
                total_count += labels.size(0)
                # correct_count += sum(1 for i in range(len(max_output)) if max_output[i] == labels[i])
                correct_count += (max_output == labels).sum().item()

        val_accuracy = 100 * correct_count / total_count
        print("Validation Accuracy (after epoch " + str(n+1) +"): " + str(val_accuracy))
        print("Predicted class distribution:", Counter(pred_counts))
        
        
    # print("Training Over. Validation beginning...")
    total_train_time = time.time() - start_time
    print("Training over after ", total_train_time, "seconds")
    if(save): torch.save(model.state_dict(), 'bird_model_weights.pth')
    return model


def load_model(): #loads model from 'bird_model_weights.pth' and returns
    model = CNN()
    model.load_state_dict(torch.load('bird_model_weights.pth'))
    return model

def test_model(model, testloader): #runs validation on model given valloader
    model.eval()
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in testloader: 
            outputs = model(images)
            _ , max_output = torch.max(outputs, 1)
            total_count += labels.size(0)
            # correct_count += sum(1 for i in range(len(max_output)) if max_output[i] == labels[i])
            correct_count += (max_output == labels).sum().item()


    accuracy = 100 * correct_count / total_count
    print("Test Accuracy: " + str(accuracy))

def main():
    # model = load_model()
    model = train_model(model=None, epochs=5, save=True)
    test_model(model, testloader)
    
if __name__ == '__main__': main()