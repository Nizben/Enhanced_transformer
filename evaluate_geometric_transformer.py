# evaluate_geometric_transformer.py

import torch
import torch.nn as nn
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
    num_classes = 10
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = Custom3DDataset(data_dir=data_dir, split='test', transform=None, K=K)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = EnhancedGeometricTransformer(embed_dim, num_heads, num_layers, sigma, dropout, K, num_classes).to(device)
    checkpoint_path = 'checkpoints/enhanced_geometric_transformer_epoch_50.pth'  # Adjust as needed
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print(f"Loaded model checkpoint from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Evaluating") as pbar:
            for points, neighbors, labels in test_loader:
                points, neighbors, labels = points.to(device), neighbors.to(device), labels.to(device)
                outputs = model(points, neighbors)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({'Loss': loss.item(), 'Accuracy': f"{100.*correct/total:.2f}%"})
                pbar.update(1)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f"\nTest Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
