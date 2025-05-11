import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from tqdm.auto import tqdm
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)


data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1., 2.])),
    torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1.]))
)

dataset = data_distribution.sample(torch.Size([10000]))
dataset_validation = data_distribution.sample(torch.Size([1000]))

fig, ax = plt.subplots(1, 1)
sns.histplot(dataset, bins=50, stat='density')
plt.title("True Data Distribution")
plt.show()

# from some tutorial online https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch 
TIME_STEPS = 250
BETA_START = 1e-4
BETA_END = 0.02
N_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 8e-4

# many betas for quick access
betas = torch.linspace(BETA_START, BETA_END, TIME_STEPS)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

g = torch.nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 128),
    nn.ReLU(),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

optimizer = torch.optim.Adam(g.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# train
epochs = tqdm(range(N_EPOCHS))
train_loss = []
for e in epochs:
    g.train()
    indices = torch.randperm(dataset.shape[0])
    shuffled_dataset = dataset[indices]

    for i in range(0, len(shuffled_dataset) - BATCH_SIZE, BATCH_SIZE):
        x0 = shuffled_dataset[i:i+BATCH_SIZE].unsqueeze(1)
        t = torch.randint(0, TIME_STEPS, (BATCH_SIZE,)).long()

        # q(x_t | x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*noise
        noise = torch.randn_like(x0)
        sqrt_ab = sqrt_alpha_bars[t].unsqueeze(1)
        sqrt_omb = sqrt_one_minus_alpha_bars[t].unsqueeze(1)
        xt = sqrt_ab * x0 + sqrt_omb * noise

        # Train network to predict the noise
        input_tensor = torch.cat([xt, t.unsqueeze(1) / TIME_STEPS], dim=1)
        predicted_noise = g(input_tensor)
        loss = loss_fn(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epochs.set_description(f"Epoch {e} Loss: {loss.item():.4f}")

@torch.no_grad()
def sample_reverse(g, count):
    x = torch.randn(count, 1)  # Start with pure noise

    for t in reversed(range(TIME_STEPS)):
        t_tensor = torch.full((count, 1), t / TIME_STEPS)
        input_tensor = torch.cat([x, t_tensor], dim=1)
        predicted_noise = g(input_tensor)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

        mean = coef1 * (x - coef2 * predicted_noise)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean
    return x

# Sampling and Plotting
samples = sample_reverse(g, 1000).squeeze().detach().numpy()

fig, ax = plt.subplots(1, 1)
bins = np.linspace(-10, 10, 50)
sns.kdeplot(dataset.numpy(), ax=ax, color='blue', label='True distribution', linewidth=2)
sns.histplot(samples, ax=ax, bins=bins, color='red', label='Sampled distribution', stat='density')
ax.legend()
ax.set_xlabel('Sample value')
ax.set_ylabel('Sample density')
plt.title("DDPM Sampling vs True Distribution")
plt.show()
