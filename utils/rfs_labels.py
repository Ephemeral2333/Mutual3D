import numpy as np

RFS_CLASSES = (
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 
    'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 
    'bathtub', 'otherfurniture', 'kitchen_cabinet', 'display', 
    'trash_bin', 'other_shelf', 'other_table'
)

CAD_CLASSES = (
    'table', 'chair', 'bookshelf', 'sofa', 'trash_bin', 
    'cabinet', 'display', 'bathtub'
)

NUM_RFS_CLASSES = len(RFS_CLASSES)  # 25
NUM_CAD_CLASSES = len(CAD_CLASSES)  # 8

RFS2CAD = {
    2: 5,   # cabinet -> cabinet
    4: 1,   # chair -> chair  
    5: 3,   # sofa -> sofa
    6: 0,   # table -> table
    9: 2,   # bookshelf -> bookshelf
    16: 1,  # toilet -> chair
    17: 7,  # sink -> bathtub
    18: 7,  # bathtub -> bathtub
    21: 6,  # display -> display
    22: 4,  # trash_bin -> trash_bin
}

RFS2CAD_arr = np.ones(30) * -1
for k, v in RFS2CAD.items():
    RFS2CAD_arr[k] = v 