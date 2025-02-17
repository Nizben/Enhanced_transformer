import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.enhanced_geometric_transformer import EnhancedGeometricTransformer
from datasets.custom_3d_dataset import Custom3DDataset
from tqdm import tqdm
import os

def main():
    # Hyperparameters
    data_dir = './data/ModelNet10'  # Replace with your dataset path
    embed_dim = 768
    num_heads = 12
    num_layers = 12
    sigma = 1.0
    dropout = 0.1
    K = 16
    num_classes = 10  # e.g., ModelNet10
    learning_rate = 1e-4
    batch_size = 4
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    train_dataset = Custom3DDataset(data_dir=data_dir, split='train', transform=None, K=K)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model, Loss, and Optimizer
    model = EnhancedGeometricTransformer(embed_dim, num_heads, num_layers, sigma, dropout, K, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{num_epochs}") as pbar:
            for points, neighbors, labels in train_loader:
                points, neighbors, labels = points.to(device), neighbors.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(points, neighbors)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Complete | Average Loss: {avg_loss:.4f}")

        checkpoint_path = f'checkpoints/enhanced_geometric_transformer_epoch_{epoch}.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
