import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images
    transforms.RandomCrop(32, padding=4),     # Random crop with padding
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color variations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

## Load the train_train_dataset
train_dataset = torchvision.datasets.CIFAR10(
    root= './',
    train=True,
    download=True,
    transform = torchvision.transforms.ToTensor()
)

test_dataset = torchvision.datasets.CIFAR10(
    root= './',
    train=False,
    download=True,
    transform = torchvision.transforms.ToTensor()
)

print("Size of the train_dataset: ", len(train_dataset))
print("Type of the train_dataset: ", type(train_dataset))
print("Length of Targets of the train_dataset: ", len(train_dataset.targets))

# Understanding the train_dataset structure:
# train_dataset[0] returns a TUPLE: (image_tensor, label)
# - train_dataset[0][0] = the image tensor (shape: [C, H, W] = [3, 32, 32] for CIFAR10)
# - train_dataset[0][1] = the label (integer 0-9 representing the class)

print("\n" + "="*60)
print("BREAKDOWN OF train_dataset[0] and train_dataset[0][0]:")
print("="*60)

print("\ntrain_dataset[0] = ", train_dataset[0])
print("Type of train_dataset[0]: ", type(train_dataset[0]))
print("Length of train_dataset[0]: ", len(train_dataset[0]))  # Should be 2 (image, label)

print("\ntrain_dataset[0][0] = the IMAGE TENSOR (first element of the tuple)")
print("train_dataset[0][0] shape  : ", train_dataset[0][0].shape)
print("train_dataset[0][0] type: ", type(train_dataset[0][0]))
print("Shape breakdown: [Channels, Height, Width] = [3, 32, 32]")
print("  - 3 channels: Red, Green, Blue (RGB)")
print("  - 32x32 pixels: Image dimensions")

print("\ntrain_dataset[0][1] = the LABEL (second element of the tuple)")
print("train_dataset[0][1] = ", train_dataset[0][1])
print("train_dataset[0][1] type: ", type(train_dataset[0][1]))
print("Label meaning: ", train_dataset.classes[train_dataset[0][1]])  # Convert number to class name
print("train_dataset classes: ", train_dataset.classes)


print("Size of the test_dataset: ", len(test_dataset))


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        ) # Input shape : [1, 3, 32, 32] output shape : [1, 32, 32, 32]
        self.batchnorm1 = nn.BatchNorm2d(32) # Input Shape: [1, 32, 32, 32] output shape : [1, 32, 32, 32]
        self.relu1 = nn.ReLU() # Input Shape: [1, 32, 32, 32] output shape : [1, 32, 32, 32]
        ## Now pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0
        ) # Input Shape: [1, 32, 32, 32] output shape : [1, 32, 16, 16]
        self.Conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        ) # Input Shape: [1, 32, 16, 16] output shape : [1, 64, 16, 16]
        self.relu2 = nn.ReLU() # Input Shape: [1, 64, 16, 16] output shape : [1, 64, 16, 16]
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0
        ) # Input Shape: [1, 64, 16, 16] output shape : [1, 64, 8, 8]
        self.Conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        ) # Input Shape: [1, 64, 8, 8] output shape : [1, 128, 8, 8]
        self.relu3 = nn.ReLU() # Input Shape: [1, 128, 8, 8] output shape : [1, 128, 8, 8]
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0
        ) # Input Shape: [1, 128, 8, 8] output shape : [1, 128, 4, 4]
        self.flatten = nn.Flatten() # Input Shape: [1, 128, 4, 4] output shape : [1, 128 * 4 * 4] = [1, 2048]
        self.linear1 = nn.Linear(2048, 1024) # Input Shape: [1, 2048] output shape : [1, 1024]
        self.relu4 = nn.ReLU() # Input Shape: [1, 1024] output shape : [1, 1024]
        self.dropout1 = nn.Dropout(0.2) # Input Shape: [1, 1024] output shape : [1, 1024]
        self.linear2 = nn.Linear(1024, 10) # Input Shape: [1, 1024] output shape : [1, 10]  
    def forward(self, x):
        x = self.Conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.Conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.Conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return x
    
model = ImageClassifier()
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_dataloader, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_model(model, test_dataloader, loss_fn):
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 30
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------")
    train_model(model, train_dataloader, loss_fn, optimizer)
    test_model(model, test_dataloader, loss_fn)

## Save the model
torch.save(model.state_dict(), "image_classifier.pth")

## If the model is already trained, we can load the model
model.load_state_dict(torch.load("image_classifier.pth"))
model.to(device)  # Ensure model is on the correct device
model.eval()  # Set to evaluation mode

## Test the model with 20 random images from the test dataset and print the label
random_indices = torch.randint(0, len(test_dataset), size=(20,))
test_images = [test_dataset[i][0] for i in random_indices]
test_labels = [test_dataset[i][1] for i in random_indices]
for i in range(20):
    print(f"Test Image {i+1}:")
    print(f"Label: {test_labels[i]} ({test_dataset.classes[test_labels[i]]})")
    # Move image to device and add batch dimension [C, H, W] -> [1, C, H, W]
    image = test_images[i].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image)
        predicted_label = pred.argmax(1).item()
    print(f"Predicted Label: {predicted_label} ({test_dataset.classes[predicted_label]})")
    print("-"*50)

print("Done!")

