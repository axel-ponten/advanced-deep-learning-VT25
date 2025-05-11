import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


# Hyperparameters
LEARNING_RATE = 4e-4
BATCH_SIZE = 128
N_EPOCHS = 3
IMAGE_SIZE = 28
TIME_STEPS = 1000
SAMPLING_TIMESTEPS = 250
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataloaders
transform = transforms.Compose([transforms.ToTensor()])

print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# setup
DIM = 32
DIM_MULTS = (1, 2, 5)

model = Unet(
    dim=DIM,
    dim_mults=DIM_MULTS,
    flash_attn=False,
    channels=1
).to(DEVICE)

diffusion = GaussianDiffusion(
    model,
    image_size=IMAGE_SIZE,
    timesteps=TIME_STEPS,
    sampling_timesteps=SAMPLING_TIMESTEPS
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# Training Loop
os.makedirs("samples", exist_ok=True)
epoch_losses = []

print("Starting training...")
for epoch in range(N_EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(DEVICE)

        loss = diffusion(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{N_EPOCHS}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"===> Epoch [{epoch+1}/{N_EPOCHS}] Average Loss: {avg_loss:.4f}")

    # Save samples every 1 epochs
    if (epoch + 1) % 1 == 0:
        model.eval()
        with torch.no_grad():
            samples = diffusion.sample(batch_size=16)
            torchvision.utils.save_image(samples, f"samples/epoch_{epoch+1}.png", nrow=4)

# Plot & Save Training Loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("samples/training_loss.png")
plt.close()

# Save Final Model
torch.save(model.state_dict(), "samples/final_model.pth")
print("Training complete. Model and loss plot saved in 'samples/'.")
