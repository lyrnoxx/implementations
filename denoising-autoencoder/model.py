import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download= True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size= 128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Linear(128, 64), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)
    
def add_noise(imgs, noise_factor=0.3):
    noisy = imgs+noise_factor* torch.randn_like(imgs)
    return torch.clip(noisy, 0., 1.)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=DenoisingAutoencoder().to(device)
criterion= nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs =imgs.to(device)
        noisy_imgs = add_noise(imgs)
        output = model(noisy_imgs)
        loss= criterion(output, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    avg_loss = total_loss/len(train_loader)
    losses.append(avg_loss)

plt.plot(losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.show()

model.eval()
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        noisy_imgs =add_noise(imgs)
        outputs= model(noisy_imgs)
        break

imgs =imgs.cpu()
noisy_imgs = noisy_imgs.cpu()
outputs=outputs.cpu()

fig, axs =plt.subplots(3, 10, figsize=(15,5))
for i in range(10):
    axs[0, i].imshow(imgs[i][0], cmap='gray')
    axs[1, i].imshow(noisy_imgs[i][0], cmap='gray')
    axs[2, i].imshow(outputs[i][0], cmap='gray')
    axs[0,i].axis('off')
    axs[1,i].axis('off')
    axs[2,i].axis('off')

axs[0,0].set_ylabel("original", fontsize=12)
axs[1,0].set_ylabel("Noisy", fontsize=12)
axs[2,0].set_ylabel("Denoised", fontsize=12)
plt.tight_layout()
plt.show()