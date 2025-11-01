import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
# Multi-GPU support - minimal additions
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# Multi-GPU setup - auto-detects if running with torchrun
if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
    # Running with torchrun - initialize DDP
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main_process = (rank == 0)
else:
    # Single GPU - original behavior
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank = 0
    local_rank = 0
    world_size = 1
    is_main_process = True

print(f"Using {device} device")

# ============================================================================
# MULTI-GPU EXPLANATION 
# ============================================================================
"""
Let's say you have 8 GPUs (like 8 workers in a factory). 

 KEY CONCEPTS:

1. WORLD_SIZE = How many workers (GPUs) are working together
   - If you have 8 GPUs: world_size = 8
   - If you have 1 GPU: world_size = 1
   - Think of it as: "How many people are on my team?"

2. RANK = Each worker's unique ID number
   - Worker 0, Worker 1, Worker 2, ... Worker 7
   - Rank is like a name tag: "Hi, I'm worker #3"
   - Rank 0 is special - it's the "primary process"
   - Rank can be 0, 1, 2, 3, 4, 5, 6, or 7 (if you have 8 GPUs)

3. LOCAL_RANK = Which GPU this worker uses on THIS computer
   - On a single computer with 8 GPUs: local_rank = 0, 1, 2, 3, 4, 5, 6, 7
   - It's like saying "I'm using GPU #2 on this machine"
   - Usually local_rank = rank (when you have one computer)

4. NCCL = The "language" GPUs use to talk to each other
   - Stands for: NVIDIA Collective Communications Library
   - GPUs use NCCL to share information (like sharing homework answers!)
   - Only works with NVIDIA GPUs

HOW IT WORKS:

Imagine training a model is like studying for a test:

SINGLE GPU (Old Way):
- You (one person) study ALL 50,000 images
- Takes a long time!

MULTI-GPU (New Way with 8 GPUs):
- Split the work: Each GPU studies 50,000 ÷ 8 = 6,250 images
- GPU 0 studies images 0-6,249
- GPU 1 studies images 6,250-12,499
- GPU 2 studies images 12,500-18,749
- ... and so on
- They all share what they learned (using NCCL)
- Much faster! 8x faster (almost)!

 EXAMPLE:
If you run: torchrun --nproc_per_node=8 image_classifier.py

What happens:
- torchrun creates 8 separate processes (like opening 8 terminals)
- Process 0: rank=0, local_rank=0, uses GPU 0
- Process 1: rank=1, local_rank=1, uses GPU 1
- Process 2: rank=2, local_rank=2, uses GPU 2
- ... and so on
- Each process gets different data to train on
- They all train at the same time!
- After each batch, they share their "learnings" (gradients) using NCCL
- The model gets updated based on what ALL 8 GPUs learned

 WHAT THE CODE DOES:

1. CHECKING FOR MULTI-GPU MODE:
   Code: if 'RANK' in os.environ or 'WORLD_SIZE' in os.environ:
   - Checks if environment variables RANK or WORLD_SIZE exist
   - If they exist = torchrun detected, use multi-GPU mode
   - If they don't exist = single GPU mode (normal Python run)

2. GETTING EACH PROCESS'S IDENTITY:
   Code: 
     rank = int(os.environ.get('RANK', 0))
     local_rank = int(os.environ.get('LOCAL_RANK', 0))
     world_size = int(os.environ.get('WORLD_SIZE', 1))
   - rank = "Which worker am I?" (0-7)
   - local_rank = "Which GPU do I use?" (0-7)
   - world_size = "How many workers total?" (8)

3. INITIALIZING COMMUNICATION:
   Code: dist.init_process_group(backend='nccl')
   - Sets up NCCL communication system
   - This is like turning on the walkie-talkies so GPUs can talk

4. ASSIGNING GPUs TO PROCESSES:
   Code:
     torch.cuda.set_device(local_rank)
     device = torch.device(f'cuda:{local_rank}')
   - Tells each process: "You're process #3, use GPU #3"
   - Makes sure each process uses its own GPU

5. IDENTIFYING THE MAIN PROCESS:
   Code: is_main_process = (rank == 0)
   - Checks if this is rank 0 (the primary process)
   - Only rank 0 prints messages and saves files
   - Prevents 8 copies of the same message!
"""
# ============================================================================

if is_main_process:
    print(f"\n{'='*70}")
    print("MULTI-GPU INFO:")
    print(f"{'='*70}")
    print(f"World Size (Total GPUs): {world_size}")
    print(f"My Rank (Worker ID): {rank}")
    print(f"My Local Rank (GPU Number): {local_rank}")
    print(f"Am I the Main Process?: {is_main_process}")
    print(f"Communication Backend: NCCL (NVIDIA GPUs)")
    print(f"{'='*70}\n")


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
    transform = train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root= './',
    train=False,
    download=True,
    transform = test_transform
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
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # Input Shape: [1, 3, 32, 32] output shape : [1, 32, 32, 32]
        self.bn1 = nn.BatchNorm2d(32) # Input Shape: [1, 32, 32, 32] output shape : [1, 32, 32, 32]      # BatchNorm for stable training
        self.relu1 = nn.ReLU() # Input Shape: [1, 32, 32, 32] output shape : [1, 32, 32, 32]
        self.pool1 = nn.MaxPool2d(2, 2) # Input Shape: [1, 32, 32, 32] output shape : [1, 32, 16, 16]
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Input Shape: [1, 32, 16, 16] output shape : [1, 64, 16, 16]
        self.bn2 = nn.BatchNorm2d(64) # Input Shape: [1, 64, 16, 16] output shape : [1, 64, 16, 16]
        self.relu2 = nn.ReLU() # Input Shape: [1, 64, 16, 16] output shape : [1, 64, 16, 16]
        self.pool2 = nn.MaxPool2d(2, 2) # Input Shape: [1, 64, 16, 16] output shape : [1, 64, 8, 8]
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Input Shape: [1, 64, 8, 8] output shape : [1, 128, 8, 8]
        self.bn3 = nn.BatchNorm2d(128) # Input Shape: [1, 128, 8, 8] output shape : [1, 128, 8, 8]
        self.relu3 = nn.ReLU() # Input Shape: [1, 128, 8, 8] output shape : [1, 128, 8, 8]
        self.pool3 = nn.MaxPool2d(2, 2) # Input Shape: [1, 128, 8, 8] output shape : [1, 128, 4, 4]
        
        # Fully connected layers
        self.flatten = nn.Flatten() # Input Shape: [1, 128, 4, 4] output shape : [1, 128 * 4 * 4] = [1, 2048]    
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Reduced size
        self.bn_fc1 = nn.BatchNorm1d(512) # Input Shape: [1, 512] output shape : [1, 512]
        self.relu_fc1 = nn.ReLU() # Input Shape: [1, 512] output shape : [1, 512]
        self.dropout1 = nn.Dropout(0.5)         # Dropout for regularization # Input Shape: [1, 512] output shape : [1, 512]
        
        self.fc2 = nn.Linear(512, 256) # Input Shape: [1, 512] output shape : [1, 256]
        self.bn_fc2 = nn.BatchNorm1d(256) # Input Shape: [1, 256] output shape : [1, 256]
        self.relu_fc2 = nn.ReLU() # Input Shape: [1, 256] output shape : [1, 256]
        self.dropout2 = nn.Dropout(0.5) # Input Shape: [1, 256] output shape : [1, 256]
        
        self.fc3 = nn.Linear(256, num_classes)  # NO activation here! # Input Shape: [1, 256] output shape : [1, num_classes]
    def forward(self, x):
                # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Fully connected
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)  # Raw logits, no activation! # Input Shape: [1, 256] output shape : [1, num_classes]
        return x
    
model = ImageClassifier()
print(model)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================================
# DDP (DistributedDataParallel) EXPLANATION
# ============================================================================
"""
What is DDP?
-----------
DDP = DistributedDataParallel

Think of it like this:
- Without DDP: Each GPU trains its own copy of the model separately
  → Like 8 students studying alone, not sharing notes
  
- With DDP: All GPUs train together and share what they learned
  → Like 8 students in a study group, sharing notes after each chapter
  
How DDP Works:
1. Each GPU gets a copy of the model
2. Each GPU trains on different data
3. After each batch, GPUs share their "gradients" (what they learned)
4. All GPUs update their models to be the same
5. Repeat!

It's like voting:
- Each GPU: "I think the model should change this way"
- DDP: "Let's average everyone's opinion"
- Result: All GPUs have the same updated model
"""
# ============================================================================

# Multi-GPU: wrap model with DDP
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

epochs = 30

# ============================================================================
# DistributedSampler EXPLANATION
# ============================================================================
"""
What is DistributedSampler?
---------------------------
It's like a smart data splitter!

Without DistributedSampler (Single GPU):
- You get ALL 50,000 training images
- You process them all yourself

With DistributedSampler (Multi-GPU with 8 GPUs):
- GPU 0 gets images: 0, 8, 16, 24, 32, ... (every 8th image starting from 0)
- GPU 1 gets images: 1, 9, 17, 25, 33, ... (every 8th image starting from 1)
- GPU 2 gets images: 2, 10, 18, 26, 34, ... (every 8th image starting from 2)
- ... and so on

Think of it like dealing cards:
- Deck of 50,000 cards (images)
- 8 players (GPUs)
- DistributedSampler deals cards so each player gets different cards
- No duplicate cards! Everyone gets a fair share.

Key Parameters:
- num_replicas=world_size: "I have 8 players"
- rank=rank: "I'm player #3"
- shuffle=True: "Shuffle the deck each epoch"

Why shuffle=False in DataLoader?
- Because DistributedSampler handles shuffling!
- If both shuffle, you'd shuffle twice (confusing!)
"""
# ============================================================================

# Multi-GPU: use DistributedSampler to split data across GPUs
if world_size > 1:
    # Create a sampler that splits data across all GPUs
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # Use the sampler (don't shuffle in DataLoader - sampler does it!)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False, sampler=train_sampler)
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
