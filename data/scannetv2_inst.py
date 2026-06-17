from __future__ import annotations
import glob
import math
import os.path as osp
import pickle
import sys
from typing import Dict, Sequence, Tuple, Union, Any, List
import numpy as np
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset

from utils.consts import MEAN_COLOR_RGB

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from model.maft.utils import Instances3D
from utils.util.bbox import BBoxUtils

BBox = BBoxUtils()


class ScanNetDataset(Dataset):
    def __init__(self,
                 data_root: str,
                 prefix: str,
                 voxel_cfg: Dict[str, Any] | None = None ,
                 training: bool = True,
                 mode: int = 4,
                 with_elastic: bool = True,
                 logger: Any | None = None,
                 exclude_zero_gt: bool = False,
                 num_classes: int = 25,
                 bbox_root: str | None = None):
        super().__init__()
        self.data_root = data_root
        self.prefix = prefix
        self.voxel_cfg = voxel_cfg or {}
        self.training = training
        self.mode = mode
        self.with_elastic = with_elastic
        self.logger = logger
        self.exclude_zero_gt = exclude_zero_gt
        self.num_classes = num_classes
        self.bbox_root = bbox_root
        self.filenames = self.get_filenames()

    def get_filenames(self) -> List[str]:
        split_file = osp.join(self.data_root, 'splits', f'{self.prefix}.txt')
        if not osp.exists(split_file):
            raise RuntimeError(f'Split file {split_file} not found.')

        with open(split_file, 'r') as f:
            scan_ids = [ln.strip() for ln in f]
        filenames = []
        for sid in scan_ids:
            data_npz_path = osp.join(self.data_root, 'processed_data', sid, 'data.npz')
            filenames.append(data_npz_path)
        return filenames

    def get_scene_index(self, scene_id: str) -> int:
        for i, filename in enumerate(self.filenames):
            if scene_id in filename:
                return i
        raise ValueError(f"场景ID {scene_id} 在数据集中未找到")
        for i, filename in enumerate(self.filenames):
            if scene_id in filename:
                return i
        raise ValueError(f'场景ID {scene_id} 在数据集中未找到')

    def load(self, filename: str):
        scan_id = osp.basename(osp.dirname(filename))
        data = np.load(filename)
        mesh_vertices = data['mesh_vertices'].astype(np.float32)
        semantic_labels = data['semantic_labels']
        instance_labels = data['instance_labels']

        xyz = mesh_vertices[:, :3]
        rgb = (mesh_vertices[:, 3:] - MEAN_COLOR_RGB) / 256.0

        semantic_label = semantic_labels.astype(np.int32)
        semantic_label[semantic_label == 255] = -100

        instance_label = instance_labels.astype(np.float32) - 1
        instance_label[instance_label == -1] = -100

        superpoint = data['superpoint_labels'].astype(np.int32)
        normal = None
        data.close()
        bboxes = None
        bbox2inst = None
        bbox_file = osp.join("datasets/scannet/processed_data", scan_id, 'bbox.pkl')
        with open(bbox_file, 'rb') as f:
            bbox_info = pickle.load(f)

        bbox2inst_list = []
        bboxes_list = []
        for item in bbox_info:
            bbox2inst_list.append(item['instance_id'] - 1)
            bboxes_list.append(item['box3D'])

        if len(bboxes_list) > 0:
            bboxes = np.stack(bboxes_list, axis=0)
            bbox2inst = np.array(bbox2inst_list)
        return xyz, rgb, superpoint, semantic_label, instance_label, normal, bboxes, bbox2inst

    def transform_train(self, xyz, rgb, superpoint, semantic_label,
                        instance_label, normal=None, bboxes=None, bbox2inst=None):
        if bboxes is not None:
            xyz_middle, bboxes_aug, normal = self.data_aug(xyz, bboxes, True, True, True, normal)
        else:
            xyz_middle, normal = self.data_aug(xyz, None, True, True, True, normal)
            bboxes_aug = None

        rgb = rgb + np.random.randn(3) * 0.1

        xyz = xyz_middle * self.voxel_cfg['scale']

        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)

        xyz_offset = xyz.min(0)
        xyz -= xyz_offset

        xyz, valid_idxs = self.crop(xyz)

        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label, instance_remap = self.get_cropped_inst_label(instance_label, valid_idxs)
        inst2bbox = {}
        for ii, ori_inst in enumerate(bbox2inst):
            if ori_inst in instance_remap:
                inst2bbox[instance_remap[ori_inst]] = ii

        return (xyz, xyz_middle, rgb, superpoint,
                semantic_label, instance_label, normal,
                bboxes_aug, bbox2inst, inst2bbox)

    def transform_test(self, xyz, rgb, superpoint, semantic_label=None,
                       instance_label=None, normal=None, bboxes=None, bbox2inst=None):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg['scale']
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs],
                               return_inverse=True)[1]
        if instance_label is not None:
            instance_label, remap = self.get_cropped_inst_label(instance_label,
                                                         valid_idxs)
        inst2bbox = {}
        for ii, ori_inst in enumerate(bbox2inst):
            if ori_inst in remap:
                inst2bbox[remap[ori_inst]] = ii
        return (xyz, xyz_middle, rgb, superpoint,
                semantic_label, instance_label, normal,
                bboxes, bbox2inst, inst2bbox)

    def data_aug(self, xyz, bboxes=None, jitter=False, flip=False, rot=False,
                 normal=None):
        m = np.eye(3)
        flip_applied = False
        rot_angle = 0

        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            flip_applied = np.random.randint(0, 2) == 1
            if flip_applied:
                m[0][0] *= -1
        if rot:
            rot_angle = np.random.rand() * 2 * math.pi
            m = m @ np.array([[math.cos(rot_angle), math.sin(rot_angle), 0],
                              [-math.sin(rot_angle), math.cos(rot_angle), 0],
                              [0, 0, 1]])

        xyz_aug = xyz @ m
        if normal is not None:
            normal = normal @ m

        if bboxes is not None:
            bboxes_aug = bboxes.copy()
            bboxes_aug[:, 0:3] = bboxes_aug[:, 0:3] @ m

            if flip_applied:
                bboxes_aug[:, 6] = np.sign(bboxes_aug[:, 6]) * np.pi - bboxes_aug[:, 6]
            if rot:
                bboxes_aug[:, 6] += rot_angle
                bboxes_aug[:, 6] = np.mod(bboxes_aug[:, 6] + np.pi, 2 * np.pi) - np.pi

            return xyz_aug, bboxes_aug, normal
        else:
            return xyz_aug, normal

    def crop(self, xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.voxel_cfg['spatial_shape'][1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg['max_npoint']:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = ((xyz_offset.min(1) >= 0) *
                          ((xyz_offset < full_scale).sum(1) == 3))
            full_scale[:2] -= 32
        return xyz_offset, valid_idxs

    def elastic(self, xyz, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(*bb).astype('float32') for _ in range(3)]
        for blur in (blur0, blur1, blur2, blur0, blur1, blur2):
            noise = [ndimage.filters.convolve(n, blur, mode='constant',
                                              cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0,
                                                      fill_value=0)
                  for n in noise]
        return xyz + np.hstack([i(xyz)[:, None] for i in interp]) * mag

    def get_cropped_inst_label(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        remap = {}
        unique_ids = np.unique(instance_label[instance_label != -100])
        for new_id, old_id in enumerate(unique_ids):
            remap[old_id] = new_id
            instance_label[instance_label == old_id] = new_id
        return instance_label, remap

    def get_instance3D(self, instance_label, semantic_label, superpoint, coord_float,
                       scan_id, bboxes=None, bbox2inst=None, inst2bbox=None):
        num_insts = instance_label.max().item() + 1
        num_points = len(instance_label)
        gt_masks, gt_labels = [], []
        gt_bboxes = []
        gt_bbox_valid = []

        gt_inst = torch.zeros(num_points, dtype=torch.int64)
        for i in range(num_insts):
            idx = torch.where(instance_label == i)

            instance_semantic_labels = semantic_label[idx]

            if len(instance_semantic_labels) == 0:
                continue

            valid_mask = instance_semantic_labels != -100
            if valid_mask.sum() == 0:
                continue

            valid_semantic_labels = instance_semantic_labels[valid_mask]
            unique_labels, counts = torch.unique(valid_semantic_labels, return_counts=True)

            max_count_idx = torch.argmax(counts)
            sem_id = unique_labels[max_count_idx]
            gt_mask = torch.zeros(num_points)
            gt_mask[idx] = 1
            gt_masks.append(gt_mask)
            gt_label = sem_id
            gt_labels.append(gt_label)
            gt_inst[idx] = (sem_id + 1) * 1000 + i + 1

            if i in inst2bbox:
                real_bbox = torch.from_numpy(bboxes[inst2bbox[i]]).float()
                gt_bbox = real_bbox
                gt_bbox_valid.append(True)
            else:
                instance_coords = coord_float[idx]
                if len(instance_coords) > 0:
                    min_coords = instance_coords.min(dim=0)[0]
                    max_coords = instance_coords.max(dim=0)[0]
                    center = (min_coords + max_coords) / 2.0
                    size = max_coords - min_coords
                    gt_bbox = torch.cat([center, size, torch.tensor([0.0])])
                else:
                    gt_bbox = torch.zeros(7)
                gt_bbox_valid.append(False)

            gt_bboxes.append(gt_bbox)

        if gt_masks:
            gt_masks = torch.stack(gt_masks, dim=0)
            gt_spmasks = torch_scatter.scatter_mean(gt_masks.float(), superpoint, dim=-1)
            gt_spmasks = (gt_spmasks > 0.5).float()
        else:
            gt_spmasks = torch.tensor([])
        gt_labels = torch.tensor(gt_labels)
        if len(gt_bboxes) > 0:
            gt_bboxes = torch.stack(gt_bboxes, dim=0)
        else:
            gt_bboxes = torch.tensor(gt_bboxes)

        gt_bbox_valid = torch.tensor(gt_bbox_valid) if gt_bbox_valid else torch.tensor([])

        assert gt_labels.shape[0] == gt_bboxes.shape[0] == gt_bbox_valid.shape[0]

        inst = Instances3D(num_points, gt_instances=gt_inst.numpy())
        inst.gt_labels = gt_labels.long()
        inst.gt_spmasks = gt_spmasks
        inst.gt_bboxes = gt_bboxes
        inst.gt_masks = gt_masks
        inst.gt_bbox_valid = gt_bbox_valid.bool()
        return inst

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        scan_id = osp.basename(osp.dirname(filename))

        if self.exclude_zero_gt and scan_id in {'scene0636_00', 'scene0154_00'}:
            return self.__getitem__(len(self.filenames) - 1)

        data = self.load(filename)
        data = (self.transform_train(*data) if self.training
                else self.transform_test(*data))
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, normal, bboxes, bbox2inst, inst2bbox = data

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        if semantic_label is not None:
            semantic_label = torch.from_numpy(semantic_label).long()
        else:
            semantic_label = torch.ones(xyz.shape[0]).long() * (-100)

        if instance_label is not None:
            instance_label = torch.from_numpy(instance_label).long()
        else:
            instance_label = torch.zeros(xyz.shape[0]).long()

        inst = self.get_instance3D(instance_label, semantic_label,
                                   superpoint, coord_float, scan_id, bboxes, bbox2inst, inst2bbox)

        return (scan_id, coord, coord_float, feat, superpoint,
                inst, normal, bboxes)

    def collate_fn(self, batch: Sequence[Tuple]) -> Dict[str, Any]:
        scan_ids, coords, coords_float, feats = [], [], [], []
        superpoints, insts, normals = [], [], []
        bboxes_list = []
        batch_offsets = [0]
        superpoint_bias = 0

        for i, data in enumerate(batch):
            (scan_id, coord, coord_float, feat, superpoint, inst, normal,
             bboxes) = data

            superpoint += superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.full((coord.shape[0], 1),
                                                i, dtype=torch.long),
                                     coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            insts.append(inst)
            normals.append(normal)
            bboxes_list.append(bboxes)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
        coords = torch.cat(coords, 0)
        coords_float = torch.cat(coords_float, 0)
        feats = torch.cat(feats, 0)
        superpoints = torch.cat(superpoints, 0).long()
        feats = torch.cat((feats, coords_float), dim=1)
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(),
                                self.voxel_cfg['spatial_shape'][0], None)
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(
            coords, len(batch), self.mode)

        return {
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'insts': insts,
            'coords_float': coords_float,
            'bboxes': bboxes_list
        }
