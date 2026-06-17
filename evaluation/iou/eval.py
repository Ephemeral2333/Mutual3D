# offline mesh IoU eval
# 2 modifications by Qiao
# extract_label, 3 or 5
# pred_files, replace
import open3d as o3d
import numpy as np
import os
import glob
import trimesh

from metrics import *

def extract_label(f, is_gt=False):
    # GT格式: scene0011_00_0_table_1.ply -> 类别在索引3
    # Pred格式(TIMR): scene0011_00_cabinet_0.2545.ply -> 类别在索引2
    # Pred格式(JIMR): scene0011_00_0_gt_1_chair_0.5657.ply -> 类别在score前
    parts = f[:-4].split('_')
    if is_gt:
        clsname = parts[3]  # GT文件: scene0011_00_0_table_1.ply
    else:
        # score是最后一个字段(float)，类别在score前，trash_bin跨两个字段
        # 先尝试 parts[-2] 是否为已知类别，若不是则尝试 parts[-3]+'_'+parts[-2]
        if parts[-2] in CAD_labels:
            clsname = parts[-2]
        elif len(parts) >= 3 and parts[-3] + '_' + parts[-2] in CAD_labels:
            clsname = parts[-3] + '_' + parts[-2]
        else:
            return None
    if clsname == 'trash': clsname = 'trash_bin'
    # 如果类别不在评估列表中，返回 None
    if clsname not in CAD_labels:
        return None
    return CAD_labels.index(clsname)


def extract_score(f):
    # scene0652_00_cabinet_0.1308.ply --> 0.1308
    return float(f[:-4].split('_')[-1])


def eval(gt_dir, pred_dir, voxel_size, threshs=[0.25, 0.5]):
    
    log_file = open(os.path.join(os.path.dirname(pred_dir), 'eval_log_iou.txt'), 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    # collect meshes (ply)
    # gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.ply')))
    # # pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))
    # pred_files = sorted(glob.glob(os.path.join(pred_dir.replace('[', '?').replace(']', '?'), '*.ply')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.ply'))) # scene0606_02
    # pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir.replace('[', '?').replace(']', '?'), '*.ply')))

    data = {}

    for f in gt_files:
        scene_name = os.path.basename(f)[:12]
        if scene_name not in data: 
            data[scene_name] = {}
        if 'gt' not in data[scene_name]: 
            data[scene_name]['gt'] = []
        if 'pred' not in data[scene_name]: 
            data[scene_name]['pred'] = []
        data[scene_name]['gt'].append(f)

    for f in pred_files:
        scene_name = os.path.basename(f)[:12]
        data[scene_name]['pred'].append(f)

    scene_names = data.keys()
    #scene_names = ['scene0011_00']

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):

        # gt mesh
        gt_files_scene = data[scene_name]['gt']

        gt_meshes = []
        gt_labels = []
        gt_bboxes = []
        for f in gt_files_scene:
            label = extract_label(os.path.basename(f), is_gt=True)
            if label is None:  # 跳过不在评估列表中的类别
                continue
            mesh = trimesh.load(f, process=False)
            
            # mesh expansion !!!!!!
            #mesh.vertices = mesh.vertices + 0.01 * mesh.vertex_normals

            gt_meshes.append(mesh)
            gt_bboxes.append(mesh.bounds.reshape(-1))
            gt_labels.append(label)
        
        # 如果没有符合评估类别的 GT，跳过此场景
        if len(gt_meshes) == 0:
            print(f'[step {sid}/{len(scene_names)}] {scene_name} skipped (no GT in eval categories)')
            continue
            
        gt_valid_mask = np.ones(len(gt_meshes))[None, :] # [B, M]
        gt_labels = np.array(gt_labels)[None, :] # [B, M], in CAD ids [0, 7]
        gt_meshes = [gt_meshes] # [B, M]
        gt_bboxes = np.array(gt_bboxes)[None, :] # [B, M, 6]

        info_mesh_gts = batched_prepare_gt(gt_valid_mask, gt_labels, gt_bboxes, gt_meshes, voxel_size=voxel_size)


        # pred mesh
        pred_files_scene = data[scene_name]['pred']
        pred_meshes = []
        pred_labels = []
        pred_bboxes = []
        pred_scores = []
        for f in pred_files_scene:
            label = extract_label(os.path.basename(f), is_gt=False)
            if label is None:  # 跳过不在评估列表中的类别
                continue
            score = extract_score(os.path.basename(f))
            mesh = trimesh.load(f, process=False)

            pred_labels.append(label)
            pred_scores.append(score)
            pred_meshes.append(mesh)
            pred_bboxes.append(mesh.bounds.reshape(-1))
        
        pred_valid_mask = np.ones(len(pred_meshes))[None, :] # [B, M]
        pred_labels = np.array(pred_labels)[None, :] # [B, M], in CAD ids [0, 7]
        pred_meshes = [pred_meshes] # [B, M]
        pred_bboxes = np.array(pred_bboxes)[None, :] # [B, M, 6]
        pred_scores = np.array(pred_scores)[None, :] # [B, M]

        info_mesh_preds = batched_prepare_pred(pred_valid_mask, pred_labels, pred_bboxes, pred_scores, pred_meshes, voxel_size=voxel_size)

        # record
        for calc in ap_calculator_list:
            calc.step(info_mesh_preds, info_mesh_gts)

        print(f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(gt_files_scene)},  #pred = {len(pred_files_scene)}')

    # output
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- IoU thresh = {threshs[i]} -----')
        print(f'----- IoU thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            if 'Q_mesh' in k: continue
            if 'mesh' not in k: continue
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file)
    
    log_file.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str)
    parser.add_argument('pred_dir', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.047)
    args = parser.parse_args()

    eval(args.gt_dir, args.pred_dir, args.voxel_size)


