# offline mesh IoU eval
# 2 modifications by Qiao
# extract_label, 3 or 5
# pred_files, replace
import open3d as o3d
import numpy as np
import os
import glob
import trimesh
import gc

from metrics import *

from lfd import MeshEncoder

def extract_label(f, is_gt=False):
    # GT格式: scene0011_00_0_table_1.ply -> 类别在索引3
    # Pred格式: scene0011_00_cabinet_0.2545.ply -> 类别在索引2
    parts = f[:-4].split('_')
    if is_gt:
        clsname = parts[3]  # GT文件: scene0011_00_0_table_1.ply
    else:
        clsname = parts[2]  # Pred文件: scene0011_00_cabinet_0.2545.ply
    if clsname == 'trash': clsname = 'trash_bin'
    # 如果类别不在评估列表中，返回 None
    if clsname not in CAD_labels:
        return None
    return CAD_labels.index(clsname)

def extract_score(f):
    # scene0652_00_cabinet_0.1308.ply --> 0.1308
    return float(f[:-4].split('_')[-1])



def eval(gt_dir, pred_dir, threshs):
    
    log_file = open(os.path.join(os.path.dirname(pred_dir), 'eval_log_lfd.txt'), 'a')

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    # collect meshes (ply)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.ply')))
    pred_files = sorted(glob.glob(os.path.join(pred_dir.replace('[', '?').replace(']', '?'), '*.ply')))

    data = {}

    for f in gt_files:
        scene_name = os.path.basename(f)[:12]
        if scene_name not in data: data[scene_name] = {}
        if 'gt' not in data[scene_name]: data[scene_name]['gt'] = []
        if 'pred' not in data[scene_name]: data[scene_name]['pred'] = []
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
        info_mesh_gts = []
        for f in gt_files_scene:
            label = extract_label(os.path.basename(f), is_gt=True)
            if label is None:  # 跳过不在评估列表中的类别
                continue
            
            mesh = MeshEncoder(trimesh.load(f, process=False), './cache_gt', os.path.basename(f)[:-4])
            #mesh = MeshEncoder(trimesh.load(f, process=False))
            mesh.align_mesh(verbose=True)

            info_mesh_gts.append((label, mesh))


        # pred mesh
        pred_files_scene = data[scene_name]['pred']
        info_mesh_preds = []
        for f in pred_files_scene:
            label = extract_label(os.path.basename(f), is_gt=False)
            if label is None:  # 跳过不在评估列表中的类别
                continue
            
            mesh = MeshEncoder(trimesh.load(f, process=False), f'./cache_pred_{pred_dir.split("/")[-2]}', os.path.basename(f)[:-4])
            #mesh = MeshEncoder(trimesh.load(f, process=False))
            mesh.align_mesh(verbose=True)

            score = extract_score(os.path.basename(f))
            # label = extract_label_gt(os.path.basename(f))
            # score = 1
            info_mesh_preds.append((label, mesh, score))

        # record
        for calc in ap_calculator_list:
            calc.step(info_mesh_preds, info_mesh_gts)

        print(f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(gt_files_scene)},  #pred = {len(pred_files_scene)}')
        
        # 释放内存：删除当前场景的 mesh 对象
        for _, mesh in info_mesh_gts:
            del mesh
        for _, mesh, _ in info_mesh_preds:
            del mesh
        del info_mesh_gts
        del info_mesh_preds
        gc.collect()
        print(f'[step {sid}] Memory released after scene {scene_name}')

    # output
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- thresh = {threshs[i]} -----')
        print(f'----- thresh = {threshs[i]} -----', file=log_file)
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
    
    args = parser.parse_args()

    eval(args.gt_dir, args.pred_dir, threshs=[2500, 5000])


