import torch
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

class Custom3DDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, K=16):
        """
        Custom dataset for 3D point clouds.
        Args:
            data_dir (str): Root directory containing point cloud files organized by class.
            split (str): 'train' or 'test'.
            transform (callable, optional): Transformation to apply to each point cloud.
            K (int): Number of nearest neighbors.
        """
        super(Custom3DDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.K = K

        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.file_paths = []
        self.labels = []

        split_dir = os.path.join(data_dir, split)
        for cls in self.classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.endswith('.ply') or file.endswith('.pcd'):
                    self.file_paths.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)  # (N, 3)
        if self.transform:
            points = self.transform(points)
        points = torch.tensor(points, dtype=torch.float32)  # (N, 3)

        # Handle point clouds with fewer than K points
        if points.shape[0] < self.K:
            pad_size = self.K - points.shape[0]
            pad_points = points[:pad_size].clone()
            points = torch.cat([points, pad_points], dim=0)

        nbrs = NearestNeighbors(n_neighbors=self.K, algorithm='auto').fit(points.numpy())
        _, indices = nbrs.kneighbors(points.numpy())
        neighbors = torch.tensor(indices, dtype=torch.long)  # (N, K)

        return points, neighbors, label
