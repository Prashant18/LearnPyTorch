import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os


## Setting up multiple GPUs
if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main_process = (rank == 0)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 0
    local_rank = 0
    world_size = 1
    is_main_process = True

print(f"Using {device} device")

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Slight rotation instead of flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=train_transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

print("Size of the train_dataset: ", len(train_dataset))
print("Type of the train_dataset: ", type(train_dataset))
print("Length of Targets of the train_dataset: ", len(train_dataset.targets))

print("Size of the test_dataset: ", len(test_dataset))
print("Type of the test_dataset: ", type(test_dataset))
print("Length of Targets of the test_dataset: ", len(test_dataset.targets))


# ------------------------------------------------------------
# Simple & Configurable ResNet for MNIST (Residual Blocks)
# ------------------------------------------------------------
# This is a ResNet-style network with residual connections.
# It is tiny, clean, and you can make it as deep as you want with ONE number.
# No repeated code, no complicated stages — just change num_blocks!

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x                  
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = F.relu(x)
        return x
        

class ResNetMNIST(nn.Module):
    def __init__(self, num_blocks=8, channels=64, num_classes=10):
        """
        num_blocks → how deep you want it (4 = very small, 8–20 = strong for MNIST)
        channels   → how wide (64 is perfect, 128 if you want overkill)
        """
        super().__init__()
        
        # First layer (like in real ResNet)
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        
        # Stack of identical residual blocks — this is where the magic happens
        # Change num_blocks and you instantly get a deeper network!
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # works for any input size
        self.fc      = nn.Linear(channels, num_classes)
        
        # Good practice: initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))   # → (batch, channels, 28, 28)
        x = self.blocks(x)                    # → same size, but much smarter features
        x = self.avgpool(x)                   # → (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)             # flatten
        x = self.fc(x)
        return x


# ------------------------------------------------------------
# Training the model
# ------------------------------------------------------------
model = ResNetMNIST(num_blocks=20, channels=64).to(device)  
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
if world_size > 1:
    # Wrap model with DDP - this makes all GPUs work together
    # device_ids=[local_rank] tells DDP which GPU this process uses
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if is_main_process:
        print("✓ Model wrapped with DDP - all GPUs will work together!\n")

def train_model(model, train_loader, loss_fn, optimizer):
    size = len(train_loader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_model(model, test_dataloader, loss_fn):
    # Only main process evaluates to avoid redundant computation
    if not is_main_process:
        return
    
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

epochs = 50


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

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

## Try to load existing checkpoint before training
# Multi-GPU: only load from main process to avoid file locking issues
checkpoint_loaded = False
if is_main_process:
    checkpoint_path = "resnet_mnist.pth"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            model_to_load = model.module if isinstance(model, DDP) else model
            
            # Check if checkpoint keys match model structure
            model_keys = set(model_to_load.state_dict().keys())
            checkpoint_keys = set(checkpoint.keys())
            
            # Check for critical mismatches (size mismatches will cause errors)
            model_state = model_to_load.state_dict()
            incompatible = False
            
            # Check if key structures are similar
            if len(model_keys) != len(checkpoint_keys):
                incompatible = True
            else:
                # Check for size mismatches in matching keys
                for key in model_keys:
                    if key in checkpoint_keys:
                        if model_state[key].shape != checkpoint[key].shape:
                            incompatible = True
                            break
            
            if incompatible or model_keys != checkpoint_keys:
                print(f"⚠️  Warning: Checkpoint structure doesn't match model architecture.")
                print(f"   Model expects {len(model_keys)} parameters, checkpoint has {len(checkpoint_keys)} parameters.")
                print("   Training from scratch...\n")
            else:
                # Safe to load - structures match
                model_to_load.load_state_dict(checkpoint, strict=True)
                checkpoint_loaded = True
                print("✓ Successfully loaded checkpoint - skipping training!\n")
        except RuntimeError as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print("   Training from scratch...\n")
        except Exception as e:
            print(f"⚠️  Unexpected error loading checkpoint: {e}")
            print("   Training from scratch...\n")
    else:
        print(f"ℹ️  No checkpoint found. Training from scratch...\n")

# Synchronize all processes - make sure checkpoint is loaded before training
if world_size > 1:
    dist.barrier()

# Only train if checkpoint wasn't loaded
if not checkpoint_loaded:
    for t in range(epochs):
        # Multi-GPU: set epoch for sampler to ensure different shuffle each epoch
        if world_size > 1:
            train_sampler.set_epoch(t)
        if is_main_process:
            print(f"Epoch {t+1}\n------------------------------")
        train_model(model, train_dataloader, loss_fn, optimizer)
        test_model(model, test_dataloader, loss_fn)

    ## Save the model after training
    # Multi-GPU: unwrap DDP before saving, only save from main process
    if is_main_process:
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), "resnet_mnist.pth")
        else:
            torch.save(model.state_dict(), "resnet_mnist.pth")
        print("✓ Model saved to resnet_mnist.pth\n")
else:
    # Model was loaded from checkpoint, just evaluate it
    if is_main_process:
        print("Evaluating loaded model:")
    test_model(model, test_dataloader, loss_fn)

# Set evaluation mode on all processes
model.eval()

## Test the model with 20 random images from the test dataset and print the label
# Only run inference from main process
if is_main_process:
    torch.manual_seed(42)  # For reproducible sampling
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
