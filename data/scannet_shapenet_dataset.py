#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from tqdm import tqdm


class ScanNetShapeNetPairDataset(Dataset):

    def __init__(
        self,
        pairs_dir: str,
        category: str = 'chair',
        num_sdf_samples: int = 16000,
        num_pc_points: int = 1024,
        shapenet_mesh_dir: str = None,
        split: str = 'train',
        train_ratio: float = 0.9,
        use_augmentation: bool = True,
        return_shapenet_pc: bool = False,
    ):
        super().__init__()

        self.pairs_dir = pairs_dir
        self.category = category
        self.num_sdf_samples = num_sdf_samples
        self.num_pc_points = num_pc_points
        self.shapenet_mesh_dir = shapenet_mesh_dir
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.return_shapenet_pc = return_shapenet_pc

        self._load_data(train_ratio)

    def _load_data(self, train_ratio):
        category_dir = os.path.join(self.pairs_dir, self.category)

        train_file = os.path.join(category_dir, 'train_pairs.npz')
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data not found: {train_file}")

        data = np.load(train_file, allow_pickle=True)
        self.scannet_pcs = data['scannet_pcs']
        self.shapenet_pcs = data['shapenet_pcs']
        self.metadata = data['metadata']

        n_total = len(self.scannet_pcs)
        n_train = int(n_total * train_ratio)

        np.random.seed(42)
        indices = np.random.permutation(n_total)

        if self.split == 'train':
            self.indices = indices[:n_train]
        else:
            self.indices = indices[n_train:]

        print(f"Loaded {len(self.indices)} samples for {self.split} split")
        print(f"  Category: {self.category}")
        print(f"  ScanNet PC shape: {self.scannet_pcs.shape}")
        print(f"  ShapeNet PC shape: {self.shapenet_pcs.shape}")

        self.shapenet_meshes = {}
        if self.shapenet_mesh_dir is not None:
            self._preload_shapenet_meshes()

    def _preload_shapenet_meshes(self):
        print("Preloading ShapeNet meshes...")

        unique_models = set()
        for idx in self.indices:
            meta = self.metadata[idx]
            model_key = (meta['shapenet_catid'], meta['shapenet_modelid'])
            unique_models.add(model_key)

        for catid, modelid in tqdm(unique_models, desc="Loading meshes"):
            mesh_path = os.path.join(
                self.shapenet_mesh_dir,
                'watertight_scaled_simplified',
                catid,
                f'{modelid}.off'
            )
            if os.path.exists(mesh_path):
                try:
                    mesh = trimesh.load(mesh_path, force='mesh')
                    self.shapenet_meshes[(catid, modelid)] = mesh
                except Exception as e:
                    print(f"Warning: Failed to load mesh {mesh_path}: {e}")

        print(f"Loaded {len(self.shapenet_meshes)} unique meshes")

    def _sample_sdf_from_mesh(self, mesh, num_samples):
        n_surface = num_samples // 2
        n_random = num_samples - n_surface

        surface_points, _ = trimesh.sample.sample_surface(mesh, n_surface)
        surface_points += np.random.randn(*surface_points.shape) * 0.01

        random_points = np.random.uniform(-1, 1, (n_random, 3))

        xyz = np.concatenate([surface_points, random_points], axis=0)

        try:
            sdf = mesh.nearest.signed_distance(xyz)
        except Exception:
            _, distances, _ = mesh.nearest.on_surface(xyz)
            sdf = distances

        return xyz.astype(np.float32), sdf.astype(np.float32)

    def _sample_sdf_from_pointcloud(self, pc, num_samples):
        n_near = num_samples // 2
        n_random = num_samples - n_near

        indices = np.random.choice(pc.shape[0], n_near, replace=True)
        near_points = pc[indices] + np.random.randn(n_near, 3) * 0.05

        random_points = np.random.uniform(-1, 1, (n_random, 3))

        xyz = np.concatenate([near_points, random_points], axis=0).astype(np.float32)

        from scipy.spatial import KDTree
        tree = KDTree(pc)
        distances, _ = tree.query(xyz, k=1)

        sdf = distances.astype(np.float32)

        return xyz, sdf

    def _augment_pointcloud(self, pc):
        if np.random.rand() > 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            pc = pc @ rot_matrix

        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            pc = pc * scale

        if np.random.rand() > 0.5:
            noise = np.random.randn(*pc.shape) * 0.01
            pc = pc + noise

        return pc

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        scannet_pc = self.scannet_pcs[real_idx].copy()
        shapenet_pc = self.shapenet_pcs[real_idx].copy()
        meta = self.metadata[real_idx]

        if self.use_augmentation:
            if np.random.rand() > 0.5:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ], dtype=np.float32)
                scannet_pc = scannet_pc @ rot_matrix
                shapenet_pc = shapenet_pc @ rot_matrix

            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                scannet_pc = scannet_pc * scale
                shapenet_pc = shapenet_pc * scale

        if scannet_pc.shape[0] >= self.num_pc_points:
            pc_indices = np.random.choice(scannet_pc.shape[0], self.num_pc_points, replace=False)
        else:
            pc_indices = np.random.choice(scannet_pc.shape[0], self.num_pc_points, replace=True)
        scannet_pc = scannet_pc[pc_indices]

        if shapenet_pc.shape[0] >= self.num_pc_points:
            pc_indices = np.random.choice(shapenet_pc.shape[0], self.num_pc_points, replace=False)
        else:
            pc_indices = np.random.choice(shapenet_pc.shape[0], self.num_pc_points, replace=True)
        shapenet_pc_sampled = shapenet_pc[pc_indices]

        model_key = (meta['shapenet_catid'], meta['shapenet_modelid'])
        if model_key in self.shapenet_meshes:
            xyz, gt_sdf = self._sample_sdf_from_mesh(
                self.shapenet_meshes[model_key],
                self.num_sdf_samples
            )
        else:
            xyz, gt_sdf = self._sample_sdf_from_pointcloud(
                shapenet_pc,
                self.num_sdf_samples
            )

        result = {
            'xyz': torch.from_numpy(xyz).float(),
            'gt_sdf': torch.from_numpy(gt_sdf).float(),
            'point_cloud': torch.from_numpy(scannet_pc).float(),
        }

        if self.return_shapenet_pc:
            result['shapenet_pc'] = torch.from_numpy(shapenet_pc_sampled).float()

        return result


class ScanNetShapeNetLatentDataset(Dataset):

    def __init__(
        self,
        pairs_dir: str,
        latent_dir: str,
        category: str = 'chair',
        num_pc_points: int = 1024,
        split: str = 'train',
        train_ratio: float = 0.9,
        use_augmentation: bool = True,
    ):
        super().__init__()

        self.pairs_dir = pairs_dir
        self.latent_dir = latent_dir
        self.category = category
        self.num_pc_points = num_pc_points
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')

        self._load_data(train_ratio)

    def _load_data(self, train_ratio):
        category_dir = os.path.join(self.pairs_dir, self.category)

        train_file = os.path.join(category_dir, 'train_pairs.npz')
        data = np.load(train_file, allow_pickle=True)
        self.scannet_pcs = data['scannet_pcs']
        self.metadata = data['metadata']

        latent_file = os.path.join(self.latent_dir, self.category, 'latents.npz')
        if os.path.exists(latent_file):
            latent_data = np.load(latent_file, allow_pickle=True)
            self.latents = latent_data['latents']
            self.latent_model_ids = latent_data['model_ids']

            self.model_to_latent = {}
            for i, mid in enumerate(self.latent_model_ids):
                self.model_to_latent[mid] = self.latents[i]
        else:
            raise FileNotFoundError(f"Latent codes not found: {latent_file}")

        n_total = len(self.scannet_pcs)
        n_train = int(n_total * train_ratio)

        np.random.seed(42)
        indices = np.random.permutation(n_total)

        if self.split == 'train':
            self.indices = indices[:n_train]
        else:
            self.indices = indices[n_train:]

        valid_indices = []
        for idx in self.indices:
            meta = self.metadata[idx]
            if meta['shapenet_modelid'] in self.model_to_latent:
                valid_indices.append(idx)

        self.indices = np.array(valid_indices)
        print(f"Loaded {len(self.indices)} samples with latent codes for {self.split} split")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        scannet_pc = self.scannet_pcs[real_idx].copy()
        meta = self.metadata[real_idx]

        latent = self.model_to_latent[meta['shapenet_modelid']].copy()

        if self.use_augmentation:
            if np.random.rand() > 0.5:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rot_matrix = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ], dtype=np.float32)
                scannet_pc = scannet_pc @ rot_matrix

        if scannet_pc.shape[0] >= self.num_pc_points:
            pc_indices = np.random.choice(scannet_pc.shape[0], self.num_pc_points, replace=False)
        else:
            pc_indices = np.random.choice(scannet_pc.shape[0], self.num_pc_points, replace=True)
        scannet_pc = scannet_pc[pc_indices]

        return {
            'point_cloud': torch.from_numpy(scannet_pc).float(),
            'latent': torch.from_numpy(latent).float(),
        }


def collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_dir', type=str, default='data/scannet_shapenet_pairs')
    parser.add_argument('--category', type=str, default='chair')
    args = parser.parse_args()

    dataset = ScanNetShapeNetPairDataset(
        pairs_dir=args.pairs_dir,
        category=args.category,
        num_sdf_samples=1000,
        num_pc_points=1024,
        split='train',
        return_shapenet_pc=True
    )

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"xyz shape: {sample['xyz'].shape}")
    print(f"gt_sdf shape: {sample['gt_sdf'].shape}")
    print(f"point_cloud shape: {sample['point_cloud'].shape}")
    if 'shapenet_pc' in sample:
        print(f"shapenet_pc shape: {sample['shapenet_pc'].shape}")
