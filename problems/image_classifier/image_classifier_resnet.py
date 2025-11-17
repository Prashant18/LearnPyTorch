import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision

# Setting up multiple GPUS
if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main_process = (rank == 0)
else:
    # Single GPU for quick debugging
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank = 0
    local_rank = 0
    world_size = 1
    is_main_process = True

print(f"Using {device} device")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./',
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./',
    train=False,
    download=True,
    transform=test_transform
)

print("Size of the train_dataset: ", len(train_dataset))
print("Type of the train_dataset: ", type(train_dataset))
print("Length of Targets of the train_dataset: ", len(train_dataset.targets))

class ImageClassifierResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial convolution to process input image
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Input: [3, 32, 32] -> Output: [64, 32, 32]
        self.bn1 = nn.BatchNorm2d(64)  # [64, 32, 32] -> [64, 32, 32]
        self.relu = nn.ReLU(inplace=True)  # Shared ReLU
        
        # Residual Block 1 (two conv layers with skip connection)
        self.res_block1_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # [64, 32, 32] -> [64, 32, 32]
            nn.BatchNorm2d(64),  # [64, 32, 32]
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),  # [64, 32, 32] -> [64, 32, 32]
            nn.BatchNorm2d(64)   # [64, 32, 32]
        )  # Skip: add input to output, then ReLU
        
        # Residual Block 2 (with downsampling)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),  # For skip: [64, 32, 32] -> [128, 16, 16]
            nn.BatchNorm2d(128)
        )
        self.res_block2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # [64, 32, 32] -> [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # [128, 16, 16] -> [128, 16, 16]
            nn.BatchNorm2d(128)
        )  # Skip (downsampled) + output, then ReLU
        
        # Residual Block 3 (another in the same stage)
        self.res_block3_1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # [128, 16, 16] -> [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),  # [128, 16, 16] -> [128, 16, 16]
            nn.BatchNorm2d(128)
        )  # Skip: add input to output, then ReLU
        
        # Residual Block 4 (with downsampling)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),  # [128, 16, 16] -> [256, 8, 8]
            nn.BatchNorm2d(256)
        )
        self.res_block4_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # [128, 16, 16] -> [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),  # [256, 8, 8] -> [256, 8, 8]
            nn.BatchNorm2d(256)
        )  # Skip + output, then ReLU
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # [256, 8, 8] -> [256, 1, 1]
        self.fc = nn.Linear(256, num_classes)  # [256] -> [10]
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Res Block 1
        identity = x
        x = self.res_block1_1(x)
        x += identity  # Skip connection
        x = self.relu(x)
        
        # Res Block 2 (downsample)
        identity = self.downsample1(x)
        x = self.res_block2_1(x)
        x += identity
        x = self.relu(x)
        
        # Res Block 3
        identity = x
        x = self.res_block3_1(x)
        x += identity
        x = self.relu(x)
        
        # Res Block 4 (downsample)
        identity = self.downsample2(x)
        x = self.res_block4_1(x)
        x += identity
        x = self.relu(x)
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

model = ImageClassifierResNet()
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)
if world_size > 1:
    # Wrap model with DDP - this makes all GPUs work together
    # device_ids=[local_rank] tells DDP which GPU this process uses
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if is_main_process:
        print("✓ Model wrapped with DDP - all GPUs will work together!\n")

def train_model(model, train_dataloader, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0 and is_main_process:  # Only print from main process
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
    
    # Multi-GPU: aggregate results across all GPUs
    if world_size > 1:
        test_loss_tensor = torch.tensor(test_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        test_loss = test_loss_tensor.item() / world_size  # Average loss
        correct = correct_tensor.item() / world_size  # Average correct (since each GPU sees full test set)
    
    test_loss /= num_batches
    correct /= size
    if is_main_process:  # Only print from main process
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 200

# Multi-GPU: use DistributedSampler to split data across GPUs
if world_size > 1:
    # Create a sampler that splits data across all GPUs
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # Use the sampler (don't shuffle in DataLoader - sampler does it!)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=False, sampler=train_sampler)
    if is_main_process:
        print(f"✓ Using DistributedSampler - data split across {world_size} GPUs")
        print(f"  Each GPU will see {len(train_dataset) // world_size} images per epoch\n")
else:
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

for t in range(epochs):
    # Multi-GPU: set epoch for sampler to ensure different shuffle each epoch
    if world_size > 1:
        train_sampler.set_epoch(t)
    if is_main_process:
        print(f"Epoch {t+1}\n------------------------------")
    train_model(model, train_dataloader, loss_fn, optimizer)
    test_model(model, test_dataloader, loss_fn)

## Save the model
# Multi-GPU: unwrap DDP before saving, only save from main process
if is_main_process:
    if isinstance(model, DDP):
        torch.save(model.module.state_dict(), "image_classifier.pth")
    else:
        torch.save(model.state_dict(), "image_classifier.pth")

## If the model is already trained, we can load the model
# Multi-GPU: only load from main process to avoid file locking issues
# After training, the model is already trained in memory, but this section
# demonstrates how to load a saved model.
if is_main_process:
    # Only main process loads to avoid multiple processes reading the same file
    if isinstance(model, DDP):
        model.module.load_state_dict(torch.load("image_classifier.pth"))
    else:
        model.load_state_dict(torch.load("image_classifier.pth"))

# Set evaluation mode on all processes
model.eval()

## Test the model with 20 random images from the test dataset and print the label
# Only run inference from main process
if is_main_process:
    random_indices = torch.randint(0, len(test_dataset), size=(20,))
    test_images = [test_dataset[i][0] for i in random_indices]
    test_labels = [test_dataset[i][1] for i in random_indices]
    for i in range(20):
        print(f"Test Image {i+1}:")
        print(f"Label: {test_labels[i]} ({test_dataset.classes[test_labels[i]]})")
        # Move image to device and add batch dimension [C, H, W] -> [1, C, H, W]
        image = test_images[i].unsqueeze(0).to(device)
        with torch.no_grad():
            # Multi-GPU: unwrap DDP for inference
            model_to_use = model.module if isinstance(model, DDP) else model
            pred = model_to_use(image)
            predicted_label = pred.argmax(1).item()
        print(f"Predicted Label: {predicted_label} ({test_dataset.classes[predicted_label]})")
        print("-"*50)

# Multi-GPU: cleanup
if world_size > 1:
    dist.destroy_process_group()

if is_main_process:
    print("Done!")
