import pandas as pd
import numpy as np

MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673], dtype=np.float32)

RFS_labels = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'kitchen_cabinet', 'display', 'trash_bin', 'other_shelf', 'other_table']
CAD_labels = ['table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 'cabinet', 'display', 'bathtub']
CAD_cnts = [555, 1093, 212, 113, 232, 260, 191, 121]
CAD_weights = np.sum(CAD_cnts) / np.array(CAD_cnts)

CAD2ShapeNetID = ['4379243', '3001627', '2871439', '4256520', '2747177', '2933112', '3211117', '2808440']
CAD2ShapeNet = {k: v for k, v in enumerate([1, 7, 8, 13, 20, 31, 34, 43])} # selected 8 categories from SHAPENETCLASSES
ShapeNet2CAD = {v: k for k, v in CAD2ShapeNet.items()}

# cabinet, display, and bathtub (sink) may fly.
CADNotFly = [0, 1, 2, 3, 4]

# assert exist, label map file.
raw_label_map_file = 'datasets/scannet/rfs_label_map.csv'
raw_label_map = pd.read_csv(raw_label_map_file)

RFS2CAD = {} # RFS --> cad
for i in range(len(raw_label_map)):
    row = raw_label_map.iloc[i]
    RFS2CAD[int(row['rfs_ids'])] = row['cad_ids']

RFS2CAD_arr = np.ones(30) * -1
for k, v in RFS2CAD.items():
    RFS2CAD_arr[k] = v

MEAN_SIZE = {
    'bathtub': [0.5161, 0.8531, 0.4393],
    'bookshelf': [0.3379, 1.0673, 1.3376],
    'cabinet': [0.5665, 0.9601, 1.0002],
    'chair': [0.5790, 0.5515, 0.8495],
    'display': [0.1648, 0.6076, 0.4761],
    'sofa': [0.8941, 1.6924, 0.7655],
    'table': [0.7261, 1.2446, 0.6635],
    'trash_bin': [0.2788, 0.3664, 0.4561],
}

LABEL_TO_NAME = {
    3: 'cabinet', 5: 'chair', 6: 'sofa', 7: 'table',
    10: 'bookshelf', 19: 'bathtub', 22: 'display', 23: 'trash_bin',
    12: 'cabinet', 13: 'table', 15: 'cabinet', 17: 'chair',
    18: 'bathtub', 24: 'bookshelf', 25: 'table',
    25: 'table',      # other_table -> table
}

CATEGORY_COLORS = {
    'table':     [219, 94,  86],
    'chair':     [219, 194, 86],
    'bookshelf': [145, 219, 86],
    'sofa':      [86,  219, 127],
    'trash_bin': [86,  211, 219],
    'cabinet':   [86,  111, 219],
    'display':   [160, 86,  219],
    'bathtub':   [219, 86,  178],
}

LABEL_TO_CAD_IDX = {
    7: 0,    # table
    5: 1,    # chair
    10: 2,   # bookshelf
    6: 3,    # sofa
    23: 4,   # trash_bin
    3: 5,    # cabinet
    22: 6,   # display
    19: 7,   # bathtub
    13: 0,   # desk -> table
    25: 0,   # other_table -> table
    12: 5,   # counter -> cabinet
    15: 5,   # refridgerator -> cabinet
    24: 2,   # other_shelf -> bookshelf
    17: 1,   # toilet -> chair
    18: 7,   # sink -> bathtub
}