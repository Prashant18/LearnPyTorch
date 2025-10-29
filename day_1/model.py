import torch  # PyTorch: Core library for tensors, neural networks, and GPU acceleration
import torch.nn as nn  # nn: Neural network modules like layers, activations, and losses
import torch.nn.functional as F  # Functional: Stateless ops (e.g., relu without class)

# ======================================================================
# SECTION 1: DEVICE SETUP (CPU vs. GPU)
# ======================================================================
# Detect if CUDA (GPU) is available for faster training.
# Why? GPUs handle parallel computations (like matrix multiplications in neural nets) much quicker than CPUs.
# If no GPU, falls back to CPU – still works, but slower for big models/datasets.
# ASCII Visual: Think of device as your "engine":
# GPU: Turbocharged car (fast for batches of images)
# CPU: Bicycle (slower, but gets you there)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")  # Outputs "cuda" or "cpu" – check this to confirm setup

# ======================================================================
# SECTION 2: LOSS FUNCTION
# ======================================================================
# CrossEntropyLoss: Measures how wrong the model's predictions are for classification.
# Ideal for multi-class problems like FashionMNIST (10 clothing categories).
# It combines softmax (turns raw scores into probabilities) and negative log likelihood.
# Simple analogy: Penalty score – low if model guesses right with confidence, high if wrong.
# Remember from earlier: Loss = -log(prob of correct class); averages over batch.
loss_fn = nn.CrossEntropyLoss()
# Cross Entropy Loss is a loss function that is used to measure the difference between the predicted and actual labels

# ======================================================================
# SECTION 3: DATA LOADERS
# ======================================================================
# Assuming 'dataload.py' is your previous script with DataLoaders for FashionMNIST.
# It provides train_dataloader and test_dataloader: Batches of images/labels.
# Batch size (e.g., 64): Groups data for efficient processing.
# Shuffle: Randomizes training order to prevent memorizing sequence.
from dataload import train_dataloader, test_dataloader  # Import pre-made DataLoaders

# ======================================================================
# SECTION 4: DIAGNOSTIC FOR CUDA
# ======================================================================
# Quick debug print: Helps troubleshoot if GPU isn't working as expected.
# E.g., if torch.cuda.is_available() is False but you have a GPU, check drivers/CUDA version.
# Outputs a dict with versions and counts – useful for verifying setup.
# ASCII Visual: Like a car dashboard check:
# - torch: Engine version
# - cuda: Fuel type (None if no GPU)
# - cuda_available: Green light if ready
# - device_count: Number of engines (GPUs)
print({
    "torch": torch.__version__,  # PyTorch version (e.g., '2.0.1')
    "cuda": getattr(torch.version, "cuda", None),  # CUDA version if installed (e.g., '11.7')
    "cuda_available": torch.cuda.is_available(),  # True/False
    "device_count": torch.cuda.device_count(),  # Number of GPUs (usually 1 or 0)
})

# ======================================================================
# SECTION 5: MODEL ARCHITECTURE
# ======================================================================
# nn.Sequential: A simple way to stack layers like building blocks.
# Input: Flattened 28x28 image = 784 pixels.
# Layers:
# - Linear(784 -> 512): Dense layer, learns features (weights: 784x512 matrix + bias).
# - ReLU: Non-linearity – turns negative values to 0, helps learn complex patterns.
# - Linear(512 -> 10): Reduces to 10 classes.
# - ReLU: Another activation.
# - Linear(10 -> 10): Final tweak (maybe redundant, but okay for learning).
# Why ReLU? Prevents "vanishing gradients" – simple "if positive, keep; else 0".
# ASCII Visual: Data flow:
# Image (28x28) -> Flatten to [784] -> Linear(512) -> ReLU -> Linear(10) -> ReLU -> Linear(10) -> Output scores
model_arch = nn.Sequential(
    nn.Linear(28 * 28, 784),  # Linear layer: transforms input (784 pixels) to 512 features via y = xA^T + b; dimension of A is 512x784 (out_features x in_features)
    nn.ReLU(),  # Activation function: f(x) = max(0, x) – adds non-linearity
    nn.Linear(784, 10),  # Linear layer: transforms 512 features to 10 output classes via y = xA^T + b; dimension of A is 10x512 (out_features x in_features)
    nn.ReLU(),  # Activation function: f(x) = max(0, x)
    nn.Linear(10, 10),  # Linear layer: transforms 10 features to 10 output classes via y = xA^T + b; dimension of A is 10x10 (out_features x in_features)
)

# ======================================================================
# SECTION 6: MODEL CLASS DEFINITION
# ======================================================================
# Custom model class: Inherits from nn.Module (base for all PyTorch models).
# __init__: Sets up layers (flatten + the sequential stack).
# forward: Defines how data flows through (input -> flatten -> stack -> output).
# Flatten: Turns [N,1,28,28] batch into [N,784] for linear layers.
# Logits: Raw scores (not probabilities) – one per class.
# Why class? For organization; could use just Sequential, but this is more flexible.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # Call parent class init
        self.flatten = nn.Flatten()  # Flatten the input tensor to 1D (from [1,28,28] to [784])
        self.linear_relu_stack = model_arch  # Chain together multiple layers of a neural network
    def forward(self, x):  # x: Input batch [N, C, H, W] e.g., [64,1,28,28]
        x = self.flatten(x)  # Now [N, 784]
        logits = self.linear_relu_stack(x)  # Forward pass through the neural network; outputs [N,10]
        return logits

# ======================================================================
# SECTION 7: MODEL INSTANTIATION AND OPTIMIZER
# ======================================================================
# Create model instance and move to device (GPU if available).
# Print(model): Shows layer structure – great for debugging.
# Optimizer: SGD – adjusts weights to minimize loss.
# lr=1e-3 (0.001): Learning rate – step size; too big = unstable, too small = slow.
# model.parameters(): All trainable weights/biases.
# From earlier: SGD uses gradients to nudge params downhill on loss landscape.
model = NeuralNetwork().to(device)  # .to(device): Moves model to GPU/CPU
print(model)  # Outputs model summary (layers and params)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # SGD: Simple optimizer with learning rate

# ======================================================================
# SECTION 8: TRAINING LOOP FUNCTION
# ======================================================================
# train_loop: One pass over training data.
# model.train(): Enables training mode (e.g., for dropout if used; not here).
# For each batch: Move to device, predict, compute loss.
# Backpropagation: .backward() computes gradients (slopes).
# optimizer.step(): Updates weights using gradients.
# .zero_grad(): Clears old gradients (prevents accumulation).
# Print every 100 batches: Progress check.
# ASCII Visual: Training cycle:
# Data batch -> Forward (pred) -> Loss -> Backward (grads) -> Step (update) -> Repeat
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # Total training samples (60,000)
    model.train()  # Set model to training mode
    for batch, (X, y) in enumerate(dataloader):  # batch: Index; X: Images; y: Labels
        X, y = X.to(device), y.to(device)  # Move to GPU/CPU
        pred = model(X)  # Forward: Get logits [64,10]
        loss = loss_fn(pred, y)  # Compute cross-entropy loss

        # Backpropagation
        loss.backward()  # Compute gradients for all params
        optimizer.step()  # Update params using SGD
        optimizer.zero_grad()  # Reset gradients for next batch

        if batch % 100 == 0:  # Print progress every 100 batches
            loss, current = loss.item(), (batch + 1) * len(X)  # Current: Samples processed
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # Formatted output

# ======================================================================
# SECTION 9: TESTING FUNCTION
# ======================================================================
# test: Evaluate on test data (no training).
# model.eval(): Evaluation mode (disables training-specific ops).
# torch.no_grad(): Saves memory/compute by skipping gradient tracking.
# Accumulate loss and correct predictions.
# argmax(1): Picks class with highest logit (predicted label).
# Average loss; accuracy = correct / total.
# Print results – aim for high accuracy, low loss.
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # Total test samples (10,000)
    num_batches = len(dataloader)  # Number of batches (~157 for batch=64)
    model.eval()  # Set to eval mode
    test_loss, correct = 0, 0  # Trackers
    with torch.no_grad():  # No gradients needed (faster, less memory)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # Forward: Get logits
            test_loss += loss_fn(pred, y).item()  # Sum losses
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # Count matches
    test_loss /= num_batches  # Average loss
    correct /= size  # Accuracy fraction
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# ======================================================================
# SECTION 10: MAIN TRAINING LOOP
# ======================================================================
# epochs: Full passes over dataset (10 here – try more for better accuracy).
# For each epoch: Train, then test.
# Expect loss to decrease, accuracy to increase over epochs.
# ASCII Visual: Epochs like laps in a race:
# Lap 1: High loss, low acc
# ...
# Lap 10: Low loss, high acc (hopefully 80%+ for this simple model)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------")  # Header
    train_loop(train_dataloader, model, loss_fn, optimizer)  # Train one epoch
    test(test_dataloader, model, loss_fn)  # Evaluate
print("Done!")  # Training complete

# ======================================================================
# SECTION 11: SAVING THE MODEL
# ======================================================================
# Save trained weights (state_dict) to file.
# Load later with: model.load_state_dict(torch.load("model.pth"))
# Why save? Reuse without retraining – e.g., for inference on new images.
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
