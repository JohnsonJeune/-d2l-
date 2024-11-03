import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)  # 均值向量
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)  # 方差向量 (对数形式)

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.mu_layer(hidden)
        log_var = self.log_var_layer(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # 标准差 = exp(0.5 * log方差)
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        return mu + eps * std  # 重参数化技巧

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# 损失函数：重构损失 + KL 散度
def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')  # 重构损失
    # KL 散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# 超参数
batch_size = 128
epochs = 10
learning_rate = 1e-3
latent_dim = 20  # 潜在向量的维度

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平成向量
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(input_dim=784, hidden_dim=400, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练 VAE 模型
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item() / len(data):.4f}')
    
    print(f'====> Epoch: {epoch+1} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# 测试和生成新样本
model.eval()
with torch.no_grad():
    # 从标准正态分布中采样潜在变量 z
    z = torch.randn(64, latent_dim).to(device)
    generated = model.decode(z).cpu().view(-1, 1, 28, 28)

    # 可视化生成的样本
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(generated[i, 0], cmap='gray')
        ax.axis('off')
    plt.show()