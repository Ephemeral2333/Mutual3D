
#!/usr/bin/env python3
import argparse
import torch
import yaml
import os
import sys
import numpy as np
import trimesh
import pickle
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.maft.model.maft import MAFT
from model.maft.model.coupled_maft import CoupledMAFT
from data import build_dataset, build_dataloader
from utils.logger import get_logger
from model.maft.utils.mask_encoder import rle_decode
from utils.util.bbox import BBoxUtils
from utils.consts import LABEL_TO_CAD_IDX

bbox_utils = BBoxUtils(num_heading_bin=12)

SHAPENET_CAT_MAP = {
    '04379243': 'table',
    '03001627': 'chair',
    '02871439': 'bookshelf',
    '04256520': 'sofa',
    '02747177': 'trash_bin',
    '02933112': 'cabinet',
    '03211117': 'display',
    '02808440': 'bathtub'
}

LABEL_TO_NAME = {
    3: 'cabinet',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    10: 'bookshelf',
    19: 'bathtub',
    22: 'display',
    23: 'trash_bin',
    12: 'cabinet',
    13: 'table',
    15: 'cabinet',
    17: 'chair',
    18: 'bathtub',
    24: 'bookshelf',
    25: 'table',
}

LABEL_TO_SHAPENET = {
    3: '02933112',
    5: '03001627',
    6: '04256520',
    7: '04379243',
    10: '02871439',
    19: '02808440',
    22: '03211117',
    23: '02747177',
    12: '02933112',
    13: '04379243',
    15: '02933112',
    17: '03001627',
    18: '02808440',
    24: '02871439',
    25: '04379243',
}


def load_gt_bbox_data(scene_id, scannet_processed_dir):
    bbox_path = os.path.join(scannet_processed_dir, scene_id, 'bbox.pkl')
    if not os.path.exists(bbox_path):
        return None

    with open(bbox_path, 'rb') as f:
        bbox_data = pickle.load(f)
    return bbox_data


def normalize_shapenet_to_unit_cube(points):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2
    diag_len = np.linalg.norm(max_bound - min_bound)
    if diag_len < 1e-6:
        diag_len = 1.0
    scale = 2.0 / diag_len
    return (points - center) * scale


def load_shapenet_pointcloud(shapenet_root, category_id, model_id, num_points=2048):
    for catid in [category_id, '0' + category_id if not category_id.startswith('0') else category_id]:
        pc_path = os.path.join(shapenet_root, 'pointcloud', catid, f'{model_id}.npz')
        if os.path.exists(pc_path):
            data = np.load(pc_path)
            points = data['points']

            points = normalize_shapenet_to_unit_cube(points)

            if points.shape[0] >= num_points:
                idx = np.random.choice(points.shape[0], num_points, replace=False)
            else:
                idx = np.random.choice(points.shape[0], num_points, replace=True)

            return points[idx]
    return None


def load_gt_latent(modulations_dir, category_name, model_id):
    latent_path = os.path.join(modulations_dir, category_name, model_id, 'latent.txt')
    if os.path.exists(latent_path):
        with open(latent_path, 'r') as f:
            latent = np.array([float(x) for x in f.read().strip().split()], dtype=np.float32)
        return latent
    return None


def match_gt_by_position(pred_bbox_center, gt_bbox_list, label_id, threshold=0.5, verbose=False):
    target_shapenet_catid = LABEL_TO_SHAPENET.get(label_id)
    if target_shapenet_catid is None:
        if verbose:
            print(f"    Warning: label_id {label_id} has no corresponding ShapeNet category")
        return None

    best_match = None
    best_dist = float('inf')
    all_dists = []

    for gt_item in gt_bbox_list:
        gt_catid = gt_item.get('shapenet_catid', '')
        gt_catid_clean = gt_catid.lstrip('0')
        target_clean = target_shapenet_catid.lstrip('0')

        if gt_catid_clean != target_clean and gt_catid != target_shapenet_catid:
            continue

        gt_center = gt_item['box3D'][:3]
        dist = np.linalg.norm(pred_bbox_center - gt_center)
        all_dists.append((gt_item.get('instance_id'), dist, gt_center))

        if dist < best_dist and dist < threshold:
            best_dist = dist
            best_match = gt_item

    return best_match


def decode_bbox_angle(bbox):
    if len(bbox) == 30:
        angle_cls = bbox[6:18]
        angle_reg = bbox[18:30]
        max_cls_idx = np.argmax(angle_cls)
        angle = bbox_utils.class2angle(max_cls_idx, angle_reg[max_cls_idx])
        return angle
    elif len(bbox) >= 7:
        return bbox[6]
    return 0.0


def to_canonical(points, bbox_center, bbox_size, angle):
    pts = points.copy() - bbox_center

    cos_ = np.cos(-angle)
    sin_ = np.sin(-angle)
    x_new = pts[:, 0] * cos_ - pts[:, 1] * sin_
    y_new = pts[:, 0] * sin_ + pts[:, 1] * cos_
    pts[:, 0] = x_new
    pts[:, 1] = y_new

    canonical = np.zeros_like(pts)
    canonical[:, 0] = -pts[:, 1]
    canonical[:, 1] = pts[:, 2]
    canonical[:, 2] = -pts[:, 0]

    transform_info = {
        'bbox_center': bbox_center,
        'bbox_size': bbox_size,
        'angle': angle,
    }

    return canonical, transform_info


def from_canonical(points, transform_info):
    bbox_center = transform_info['bbox_center']
    angle = transform_info['angle']

    pts = np.zeros_like(points)
    pts[:, 0] = -points[:, 2]
    pts[:, 1] = -points[:, 0]
    pts[:, 2] = points[:, 1]

    cos_ = np.cos(angle)
    sin_ = np.sin(angle)
    x_new = pts[:, 0] * cos_ - pts[:, 1] * sin_
    y_new = pts[:, 0] * sin_ + pts[:, 1] * cos_
    pts[:, 0] = x_new
    pts[:, 1] = y_new

    pts = pts + bbox_center

    return pts


def from_canonical_mesh(mesh, transform_info):
    vertices = from_canonical(mesh.vertices.copy(), transform_info)
    mesh.vertices = vertices
    return mesh


def normalize_to_unit_cube(points):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2
    diag_len = np.linalg.norm(max_bound - min_bound)
    if diag_len < 1e-6:
        diag_len = 1.0
    scale = 2.0 / diag_len
    return (points - center) * scale, center, 1.0 / scale


def get_args():
    parser = argparse.ArgumentParser('TIMR + Diffusion-SDF multi-category testing (simplified)')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('checkpoint', type=str, help='checkpoint')
    parser.add_argument('--out', type=str, required=True, help='output directory')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--category-config', type=str,
                        default='configs/category_configs.yaml',
                        help='Multi-category Diffusion-SDF config file')

    parser.add_argument('--target-labels', nargs='+', type=int, default=None,
                        help='Label IDs to process (default: all categories in config)')
    parser.add_argument('--scene', type=str, default=None,
                        help='Target scene (e.g. scene0011_00); process all scenes if omitted')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--min-points', type=int, default=100,
                        help='Minimum points per instance')

    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples per instance (best is selected)')
    parser.add_argument('--recon-mode', choices=['diffusion', 'vae'], default='diffusion',
                        help='Reconstruction mode: diffusion=conditional diffusion sampling, vae=VAE + SDF marching cubes only')
    parser.add_argument('--mesh-resolution', type=int, default=None,
                        help='Override mesh_resolution in category config; e.g. 64 reduces mesh vertex count significantly')
    parser.add_argument('--use-gt-filter', action='store_true',
                        help='Use GT ShapeNet point cloud to select the best reconstruction')
    parser.add_argument('--gt-match-threshold', type=float, default=0.5,
                        help='Position matching threshold (meters)')

    parser.add_argument('--save-intermediate', action='store_true',
                        help='Save intermediate results (canonical pc)')

    parser.add_argument('--use-gt-bbox', action='store_true',
                        help='Use GT bbox size during denormalization (requires GT match first)')

    return parser.parse_args()


def main():
    args = get_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    logger = get_logger('TIMR-MultiCategory')

    category_config_path = args.category_config
    if not os.path.isabs(category_config_path):
        category_config_path = os.path.join(os.path.dirname(__file__), category_config_path)

    with open(category_config_path, 'r') as f:
        category_config = yaml.safe_load(f)

    scannet_processed_dir = os.path.join(os.path.dirname(__file__), 'datasets/scannet/processed_data')
    shapenet_root = category_config.get('shapenet_root', 'datasets/ShapeNetv2_data')
    modulations_dir = 'config/stage1_sdf/modulations'

    categories_dict = {int(k): v for k, v in category_config['categories'].items()}
    supported_labels = list(categories_dict.keys())

    if args.target_labels:
        target_labels = [l for l in args.target_labels if l in supported_labels]
    else:
        target_labels = supported_labels

    os.makedirs(args.out, exist_ok=True)

    mesh_dir = os.path.join(args.out, 'mesh')
    os.makedirs(mesh_dir, exist_ok=True)

    all_dir = os.path.join(args.out, 'all')
    os.makedirs(all_dir, exist_ok=True)

    origin_dir = os.path.join(args.out, 'origin')
    os.makedirs(origin_dir, exist_ok=True)

    canonical_dir = None
    if args.save_intermediate:
        canonical_dir = os.path.join(args.out, 'canonical')
        os.makedirs(canonical_dir, exist_ok=True)

    instance_seg_dir = os.path.join(args.out, 'instance_seg')
    os.makedirs(instance_seg_dir, exist_ok=True)

    model_config = cfg['model'].copy()
    model_config.pop('name', None)

    coupling_cfg = cfg.get('coupling', {})
    coupling_enabled = coupling_cfg.get('enabled', False)

    if coupling_enabled:
        logger.info("Loading CoupledMAFT model...")
        base_model = MAFT(**model_config)
        model = CoupledMAFT(
            maft_model=base_model,
            config=cfg,
            device=device,
        )
    else:
        logger.info("Loading standard MAFT model...")
        model = MAFT(**model_config)

    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    model_state = model.state_dict()
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_state[k] = v
        else:
            logger.warning(f"Skipping mismatched weight: {k}")

    model.load_state_dict(filtered_state, strict=False)
    model.eval()
    logger.info(f"Model loaded: {len(filtered_state)}/{len(state_dict)} weights loaded")

    dataset = build_dataset(cfg['data']['test'], logger)

    if args.scene:
        dataset.filenames = [f for f in dataset.filenames if args.scene in f]

    dataloader = build_dataloader(dataset, training=False, **cfg['dataloader']['test'])

    from utils.multi_category_reconstructor import MultiCategoryReconstructor

    reconstructor = MultiCategoryReconstructor(
        category_config_path=category_config_path,
        device=device,
        mesh_resolution=args.mesh_resolution
    )
    logger.info(f"Reconstruction mode: {args.recon_mode}")

    gt_bbox_cache = {}
    gt_match_stats = {'matched': 0, 'unmatched': 0}

    total_instances = 0
    total_reconstructed = 0

    progress_bar = tqdm(total=len(dataloader), desc="Processing scenes")

    with torch.no_grad():
        for batch in dataloader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            result = model(batch, mode='predict')
            scan_id = result.get('scan_id', 'unknown')

            if 'pred_instances' not in result or 'coords_float' not in batch:
                progress_bar.update()
                continue

            scene_points = batch['coords_float']
            if isinstance(scene_points, list):
                scene_points = scene_points[0]
            scene_points = scene_points.cpu().numpy()

            pred_instances = result['pred_instances']

            gt_bbox_list = None
            if args.use_gt_filter or args.use_gt_bbox:
                if scan_id not in gt_bbox_cache:
                    gt_bbox_cache[scan_id] = load_gt_bbox_data(scan_id, scannet_processed_dir)
                gt_bbox_list = gt_bbox_cache[scan_id]

            scene_instances_by_label = defaultdict(list)
            scene_instances_meta = []

            inst_count = 0
            for i, inst in enumerate(pred_instances):
                conf = inst.get('conf', 0)
                if conf < args.conf_threshold:
                    continue

                label_id = int(inst['label_id'])
                if label_id not in target_labels:
                    continue

                mask = rle_decode(inst['pred_mask'])
                if mask.shape[0] != scene_points.shape[0]:
                    length = min(mask.shape[0], scene_points.shape[0])
                    mask = mask[:length]
                    pts = scene_points[:length]
                else:
                    pts = scene_points

                inst_points = pts[mask.astype(bool)]
                if inst_points.shape[0] < args.min_points:
                    continue

                bbox = inst['bboxes']
                pred_bbox_center = bbox[:3]
                pred_bbox_size = bbox[3:6]
                pred_angle = decode_bbox_angle(bbox)

                category_name = LABEL_TO_NAME.get(label_id, f'label{label_id}')
                instance_id = f"{scan_id}_{category_name}_{conf:.4f}"

                cad_idx = LABEL_TO_CAD_IDX.get(label_id)
                if cad_idx is not None:
                    mask_indices = np.where(mask.astype(bool))[0].astype(np.int32)
                    seg_iou_filename = f"{scan_id}_{category_name}_{conf:.4f}_{inst_count}.npz"
                    np.savez(
                        os.path.join(instance_seg_dir, seg_iou_filename),
                        semantic_label=cad_idx,
                        score=conf,
                        cluster_point_idxs=mask_indices
                    )

                gt_latent = None
                gt_match_info = None
                gt_bbox_info = None
                if (args.use_gt_filter or args.use_gt_bbox) and gt_bbox_list is not None:
                    gt_match = match_gt_by_position(
                        pred_bbox_center, gt_bbox_list, label_id,
                        threshold=args.gt_match_threshold,
                        verbose=False
                    )
                    if gt_match is not None:
                        cat_name = categories_dict[label_id]['name']
                        gt_latent = load_gt_latent(
                            modulations_dir,
                            cat_name,
                            gt_match['shapenet_id']
                        )
                        gt_bbox_info = {
                            'center': gt_match['box3D'][:3],
                            'size': gt_match['box3D'][3:6],
                            'angle': gt_match['box3D'][6],
                        }
                        if gt_latent is not None:
                            gt_match_info = {
                                'shapenet_catid': gt_match['shapenet_catid'],
                                'shapenet_id': gt_match['shapenet_id'],
                                'gt_instance_id': gt_match.get('instance_id'),
                                'match_dist': float(np.linalg.norm(pred_bbox_center - gt_match['box3D'][:3]))
                            }
                            gt_match_stats['matched'] += 1
                        else:
                            gt_match_stats['unmatched'] += 1
                    else:
                        gt_match_stats['unmatched'] += 1

                if args.use_gt_bbox and gt_bbox_info is not None:
                    bbox_center = gt_bbox_info['center']
                    bbox_size = gt_bbox_info['size']
                    angle = gt_bbox_info['angle']
                else:
                    bbox_center = pred_bbox_center
                    bbox_size = pred_bbox_size
                    angle = pred_angle

                canonical_points, transform_info = to_canonical(
                    inst_points, bbox_center, bbox_size, angle
                )

                points_normalized, norm_center, norm_scale = normalize_to_unit_cube(canonical_points)

                instance_data = {
                    'instance_id': instance_id,
                    'scan_id': scan_id,
                    'label_id': label_id,
                    'conf': conf,
                    'points': points_normalized,
                    'transform_info': transform_info,
                    'norm_center': norm_center,
                    'norm_scale': norm_scale,
                    'seed': 42,
                    'gt_latent': gt_latent,
                    'gt_match_info': gt_match_info,
                    'gt_bbox_info': gt_bbox_info,
                }

                scene_instances_by_label[label_id].append(instance_data)
                scene_instances_meta.append(instance_data)

                if canonical_dir:
                    np.random.seed(label_id)
                    color = np.random.randint(0, 255, 3)
                    pc_cloud = trimesh.PointCloud(
                        vertices=canonical_points,
                        colors=np.tile(color, (canonical_points.shape[0], 1))
                    )
                    canonical_filename = f"{scan_id}_inst{inst_count}_{category_name}.ply"
                    pc_cloud.export(os.path.join(canonical_dir, canonical_filename))

                inst_count += 1

            if len(scene_instances_meta) == 0:
                progress_bar.update()
                continue

            total_instances += len(scene_instances_meta)

            reconstruction_results = reconstructor.reconstruct_batch_by_category(
                scene_instances_by_label,
                num_samples=args.num_samples,
                use_gt_filter=args.use_gt_filter,
                recon_mode=args.recon_mode,
                save_all_samples=args.save_intermediate,
                save_dir=args.out,
                progress_bar=None
            )

            scene_meshes = []

            for inst_data in scene_instances_meta:
                instance_id = inst_data['instance_id']
                label_id = inst_data['label_id']

                mesh = reconstruction_results.get(instance_id)
                if mesh is None:
                    continue

                gt_bbox_info = inst_data.get('gt_bbox_info')
                if args.use_gt_bbox and gt_bbox_info is not None:
                    bbox_size = gt_bbox_info['size']
                    transform_info_for_inverse = {
                        'bbox_center': gt_bbox_info['center'],
                        'bbox_size': gt_bbox_info['size'],
                        'angle': gt_bbox_info['angle'],
                    }
                else:
                    bbox_size = inst_data['transform_info']['bbox_size']
                    transform_info_for_inverse = inst_data['transform_info']

                mesh_min = mesh.vertices.min(axis=0)
                mesh_max = mesh.vertices.max(axis=0)
                mesh_center = (mesh_min + mesh_max) / 2
                mesh_size = mesh_max - mesh_min

                target_size_canonical = np.array([bbox_size[1], bbox_size[2], bbox_size[0]])

                scale_factors = target_size_canonical / (mesh_size + 1e-6)

                mesh.vertices = mesh.vertices - mesh_center
                mesh.vertices = mesh.vertices * scale_factors

                mesh = from_canonical_mesh(mesh, transform_info_for_inverse)

                np.random.seed(label_id)
                color = np.random.randint(0, 255, 3)
                mesh.visual.face_colors = np.array([*color, 255])

                mesh_path = os.path.join(mesh_dir, f"{instance_id}.ply")
                mesh.export(mesh_path)

                scene_meshes.append(mesh)
                total_reconstructed += 1

            scene_pc = trimesh.PointCloud(vertices=scene_points)
            scene_pc_path = os.path.join(origin_dir, f"{scan_id}.ply")
            scene_pc.export(scene_pc_path)

            if len(scene_meshes) > 0:
                combined = trimesh.util.concatenate(scene_meshes)
                combined_path = os.path.join(all_dir, f"{scan_id}_all.ply")
                combined.export(combined_path)

            del scene_instances_by_label
            del scene_instances_meta
            del reconstruction_results
            del scene_meshes
            import gc
            gc.collect()

            progress_bar.set_postfix({'scene': scan_id, 'instances': inst_count, 'reconstructed': total_reconstructed})
            progress_bar.update()

    progress_bar.close()

    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Done: {total_instances} instances total, {total_reconstructed} reconstructed successfully")


if __name__ == '__main__':
    main()
