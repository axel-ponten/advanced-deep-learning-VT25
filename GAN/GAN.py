import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
lr_gen  = 3e-3
lr_disc = 3e-3
batchSize = 32
numEpochs = 100
logStep = 625

latent_dimension = 128
image_dimension = 28 * 28 * 1

myTransforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

print("loading MNIST digits dataset")
dataset = datasets.MNIST(root="dataset/", transform=myTransforms, download=True)
loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dimension, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_dimension),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dimension, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

discriminator = Discriminator().to(device)
generator = Generator().to(device)
opt_discriminator = optim.Adam(discriminator.parameters(), lr=lr_disc)
opt_generator = optim.Adam(generator.parameters(), lr=lr_gen)

criterion = nn.BCELoss()

step = 0
fixed_noise = torch.randn(32, latent_dimension).to(device)
writer = SummaryWriter("logs")
print("Started Training and visualization...")
for epoch in range(numEpochs):
    print()
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        noise = torch.randn(batch_size, latent_dimension).to(device)
        fake = generator(noise)

        # Train Discriminator
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        disc_real_output = discriminator(real)
        loss_real = criterion(disc_real_output, real_labels)

        disc_fake_output = discriminator(fake.detach())
        loss_fake = criterion(disc_fake_output, fake_labels)

        loss_discriminator = (loss_real + loss_fake) / 2

        opt_discriminator.zero_grad()
        loss_discriminator.backward(retain_graph=True)
        opt_discriminator.step()

        # Train Generator
        gen_fake_output = discriminator(fake)
        loss_generator = criterion(gen_fake_output, real_labels)

        opt_generator.zero_grad()
        loss_generator.backward()
        opt_generator.step()

        print(f"\rEpoch [{epoch}/{numEpochs}] Batch {batch_idx}/{len(loader)} Loss discriminator: {loss_discriminator:.4f}, loss generator: {loss_generator:.4f}", end="")

        if batch_idx % logStep == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                imgGridReal = torchvision.utils.make_grid(data, normalize=True)

                writer.add_image("Fake Images", imgGridFake, global_step=step)
                writer.add_image("Real Images", imgGridReal, global_step=step)
                writer.add_scalar("Loss/Discriminator", loss_discriminator, global_step=step)
                writer.add_scalar("Loss/Generator", loss_generator, global_step=step)

        step += 1
