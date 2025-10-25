import torch  # PyTorch: Core library for tensors and neural networks
from torch.utils.data import DataLoader, Dataset  # DataLoader: For batching data; Dataset: Base class for datasets
from torchvision import datasets  # Pre-built datasets like FashionMNIST
from torchvision.transforms import ToTensor  # Transform to convert images to PyTorch tensors
import matplotlib.pyplot as plt  # For plotting visuals to see data samples

# ======================================================================
# SECTION 1: LOADING THE DATASET
# ======================================================================
# FashionMNIST is a dataset of 28x28 grayscale images of clothing items.
# It's like MNIST (handwritten digits) but for fashion: 10 classes, 60k train, 10k test images.
# Each image is a tensor: [1, 28, 28] (C=1 channel for grayscale, H=28 height, W=28 width).
# Labels are integers 0-9 corresponding to clothing types.
#
# ASCII Visual: Imagine a single image as a grid (simplified 3x3 for demo):
# +---+---+---+
# | 0 | 5 | 9 |  <- Pixel values (0=black, higher=whiter)
# +---+---+---+
# | 2 | 8 | 4 |
# +---+---+---+
# | 1 | 3 | 7 |
# +---+---+---+
# In tensor form: torch.tensor([[0,5,9], [2,8,4], [1,3,7]])
# For full 28x28, it's bigger, but same idea – each cell is a pixel intensity.

## Load the FashionMNIST training dataset
training_data = datasets.FashionMNIST(
    root="../data",  # Where to save/download data (creates folder if needed)
    train=True,      # True for training set (60,000 images)
    download=True,   # Download if not already there
    transform=ToTensor(),  # Convert images to tensors (normalizes to [0,1])
)

## Load the test dataset (similar, but 10,000 images for evaluation)
test_data = datasets.FashionMNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Print the number of training samples – helps verify loading
# Expected: 60000 – if not, check download/root path
print(len(training_data))

# ======================================================================
# SECTION 2: LABEL MAPPING
# ======================================================================
# Labels are numbers 0-9; this dict maps them to human-readable names.
# Useful for plotting or debugging – e.g., label 0 is "T-Shirt".
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# ======================================================================
# SECTION 3: VISUALIZING SAMPLES
# ======================================================================
# Why visualize? To understand data – see what images look like, check for issues.
# Here, we plot a 3x3 grid of random training images with labels.
# Uses matplotlib: Creates a figure, adds subplots, shows images in grayscale.
#
# ASCII Visual: Think of the dataset as a big stack of images:
#  _______
# | Image1 | Label: 0 (T-Shirt)
# | Image2 | Label: 3 (Dress)
# | ...    |
# | ImageN | Label: 9 (Ankle Boot)
#  -------
# Each image: A 28x28 grid of pixels, like a tiny photo.

figure = plt.figure(figsize=(8, 8))  # Create a figure (8x8 inches)
cols, rows = 3, 3  # Grid layout: 3 columns, 3 rows = 9 images
for i in range(1, cols * rows + 1):  # Loop to fill each subplot
    sample_idx = torch.randint(len(training_data), size=(1,)).item()  # Random index (0 to 59999)
    img, label = training_data[sample_idx]  # Get image tensor and label
    figure.add_subplot(rows, cols, i)  # Add a subplot at position i
    plt.title(labels_map[label])  # Set title to clothing name
    plt.axis("off")  # Hide axes for clean look
    plt.imshow(img.squeeze(), cmap="gray")  # Show image: squeeze() removes channel dim (from [1,28,28] to [28,28])
plt.savefig("data_sample.png")  # Save to file (view this PNG later to see visuals)
# Tip: Add plt.show() here if running interactively to display in a window.
# plt.show()  # Uncomment to show plot (blocks code until closed)

# ======================================================================
# SECTION 4: CREATING DATALOADERS
# ======================================================================
# DataLoader: Wraps dataset to provide batches, shuffling, etc.
# Batch size: Number of samples per batch (e.g., 64) – processes in groups for efficiency.
# Shuffle: Randomize order for better training (prevents patterns from order).
#
# ASCII Visual: Dataset without DataLoader:
# [Img1, Img2, ..., Img60000]  <- One big list, hard to handle all at once.
#
# With DataLoader (batch_size=64, shuffle=True):
# Batch1: [ImgX, ImgY, ...] (64 random images + labels)
# Batch2: [ImgA, ImgB, ...] (next 64)
# ...
# Total batches: ~938 (60000 / 64)
#
# Tensor shapes in batch:
# X (images): [N=64, C=1, H=28, W=28]  <- Stack of 64 grayscale images
# y (labels): [64]  <- List of 64 integers (0-9)
#
# Why batch? Memory efficiency, faster training with GPUs, mini-batch gradient descent.

batch_size = 64  # Common starting point; try 32 or 128 if tweaking performance
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # For training: Shuffle for randomness
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)      # For testing: Shuffle optional, but fine

# ======================================================================
# SECTION 5: INSPECTING A BATCH
# ======================================================================
# Loop through DataLoader (just once here with break).
# Prints shapes to understand structure.
# Useful for debugging: Confirm dimensions before model training.
#
# ASCII Visual: A batch tensor (simplified, N=2, C=1, H=2, W=2 for demo):
# Batch [
#   Image1: [  # Channel 1 (grayscale)
#     [0, 5],  # Row 1
#     [2, 8]   # Row 2
#   ],
#   Image2: [
#     [1, 3],
#     [7, 4]
#   ]
# ]
# In code: torch.Size([2, 1, 2, 2])  <- N=2 batches, C=1 channel, H=2 height, W=2 width

for X, y in test_dataloader:  # X: batch of images, y: batch of labels
    print(f"Shape of X [N, C, H, W]: {X.shape}")  # Expected: torch.Size([64, 1, 28, 28])
    print(f"Shape of y: {y.shape} {y.dtype}")     # Expected: torch.Size([64]) torch.int64
    break  # Stop after first batch – no need to loop all

# ======================================================================
# NEXT STEPS (FOR FUTURE YOU):
# ======================================================================
# - Build a model: Use torch.nn to create a neural net (e.g., CNN for images).
# - Train loop: Optimizer (Adam), loss (CrossEntropy), epochs.
# - Evaluate: Accuracy on test_data.
# - Experiment: Change batch_size, add transforms (e.g., Normalize).
# - Remember: Tensors are multi-dim arrays; Channels add depth (1=gray, 3=RGB).
# - Visual tip: Run this code, open 'data_sample.png' to see real images!
# - If forgetting: Search PyTorch docs or revisit this commented code.
