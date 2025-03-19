import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn.functional import F
import matplotlib.pyplot as plt


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ImageFolder(root="images/", transform=transforms)

# Create DataLoader for batch processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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
        self.conv1 = torch.nn.Conv2d(in_channels = 16, out_channels = 48, kernel_size = 3, padding = 1)

        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(48)

        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = torch.nn.Linear(48 * 28 * 28 * 28, 512)
        
        self.fc2 = torch.nn.Linear(512, 200)

    def foward(self, x):
        # Block 1
        x = self.conv1
        x = self.bn1
        x = F.relu(x)
        x = self.pool
        
        # Block 2
        x = self.conv2
        x = self.bn2
        x = F.relu(x)
        x = self.pool

        # Fully Connected Layers
        x = self.fc1
        x = F.relu(x)
        x = self.fc2
        

        return x



model = CNN()
epochs = 5
optimizer = ??
loss_func = ??

for n in range(epochs):
    run_loss = 0.0

    for images, labels in dataloader: #replace dataloader with dataloader(training)
        #forward prop
        outputs = model(images)
        loss = loss_func(outputs, labels) #define loss_func ()

        optimizer.zero_grad() #define optimizer
        loss.backward()

        optimizer.step()
        run_loss += loss.item()

    print("Epoch #" + str(n+1) + ", Loss: " + str(run_loss /len(dataloader))) #replace dataloader with dataloader(training)
print("Training Over. Validation beginning...")

correct_count = 0
total_count = 0

for images, labels in dataloader: #replace with dataloader(validate)
    ouptuts = model(images)
    _ , max_output = torch.max(outputs, 1)
    total_count += 32
    correct_count += sum(1 for i in range(len(max_output)) if max_output[i] == labels[i])

accuracy = 100 * correct_count / total_count
print("Test Accuracy: " + str(accuracy))


