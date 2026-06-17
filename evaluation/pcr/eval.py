# offline mesh IoU eval

import open3d as o3d
import numpy as np
import os
import glob
from torch import gt
import trimesh

from metrics import *

def extract_label(f):
    # Pred格式: scene0011_00_cabinet_0.2545.ply -> 类别在索引2
    parts = f[:-4].split('_')
    clsname = parts[2]
    if clsname == 'trash': clsname = 'trash_bin'
    # 如果类别不在评估列表中，返回 None
    if clsname not in CAD_labels:
        return None
    return CAD_labels.index(clsname)


def extract_score(f):
    # scene0652_00_cabinet_0.1308.ply --> 0.1308
    return float(f[:-4].split('_')[-1])

def read_txt(file):
    with open(file, 'r') as f:
        output = [x.strip() for x in f.readlines()]
    return output

def eval(pred_dir, threshs=[0.5]):

    import metrics as _metrics_module
    log_dir = os.path.dirname(pred_dir)
    log_file = open(os.path.join(log_dir, 'eval_log_pcr.txt'), 'a')
    step_log_file = open(os.path.join(log_dir, 'eval_step_log_pcr.txt'), 'a')
    _metrics_module._step_log_file = step_log_file
    _metrics_module._result_log_file = log_file

    # prepare calcs
    ap_calculator_list = [APCalculator(iou_thresh, CAD_labels) for iou_thresh in threshs]

    # collect meshes (ply)
    test_scans = read_txt(os.path.join('datasets/splits/', 'val.txt'))
    
    pred_files = sorted(glob.glob(os.path.join(pred_dir, '*.ply')))

    data = {}

    for scene_name in test_scans:
        data[scene_name] = []
        
    for f in pred_files:
        scene_name = os.path.basename(f)[:12]
        data[scene_name].append(f)

    scene_names = data.keys()
    #scene_names = ['scene0011_00']

    # loop each scene, prepare inputs
    for sid, scene_name in enumerate(scene_names):

        # gt points
        info_mesh_gts = []
        scan_data = np.load(f'datasets/scannet/processed_data/{scene_name}/data.npz')
        xyz = scan_data['mesh_vertices'].astype(np.float32)[:, :3]
        semantic_label = scan_data['semantic_labels'].astype(np.int32)
        instance_label = scan_data['instance_labels'].astype(np.int32) - 1
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_) # returns a one-element tuple, like ([0,1,29,43,...],)
            if len(inst_idx_i[0]) == 0: continue # null instance
            xyz_i = xyz[inst_idx_i] # [Ni, 3]
            label_i = semantic_label[inst_idx_i[0]][0]
            if label_i == 255: continue # ignored class
            label_i = RFS2CAD[label_i]
            if label_i == -1: continue # non-CAD class
            if label_i >= len(CAD_labels): continue  # 跳过不在评估列表中的类别
            info_mesh_gts.append((label_i, xyz_i))


        # pred mesh
        info_mesh_preds= []
        pred_files_scene = data[scene_name]
        for f in pred_files_scene:
            label = extract_label(os.path.basename(f))
            if label is None:  # 跳过不在评估列表中的类别
                continue
            mesh = trimesh.load(f, process=False)
            # 简化大型 mesh 以减少内存消耗（使用 Open3D）
            if len(mesh.faces) > 10000:
                mesh_o3d = o3d.geometry.TriangleMesh()
                mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
                mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
                mesh_o3d = mesh_o3d.simplify_quadric_decimation(10000)
                mesh = trimesh.Trimesh(
                    vertices=np.asarray(mesh_o3d.vertices),
                    faces=np.asarray(mesh_o3d.triangles)
                )
            info_mesh_preds.append((label, mesh, extract_score(os.path.basename(f))))
    
        # record
        for calc in ap_calculator_list:
            calc.step([info_mesh_preds], [info_mesh_gts])

        msg = f'[step {sid}/{len(scene_names)}] {scene_name} #gt = {len(info_mesh_gts)},  #pred = {len(info_mesh_preds)}'
        print(msg)
        print(msg, file=step_log_file, flush=True)

    # output
    print(f'===== {pred_dir} =====')
    print(f'===== {pred_dir} =====', file=log_file)
    for i, calc in enumerate(ap_calculator_list):
        print(f'----- thresh = {threshs[i]} -----')
        print(f'----- thresh = {threshs[i]} -----', file=log_file)
        metrics_dict = calc.compute_metrics()
        for k, v in metrics_dict.items():
            print(f"{k: <50}: {v}")
            print(f"{k: <50}: {v}", file=log_file, flush=True)
    
    log_file.close()
    step_log_file.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', type=str)
    args = parser.parse_args()

    eval(args.pred_dir)


