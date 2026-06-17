import os
import shutil
from pathlib import Path

def fix_scannet_structure():
    base_path = Path('data/scannetv2/scans')
    
    scene_files = list(base_path.glob('scene*_*_*.ply'))
    
    for scene_file in scene_files:
        scene_id = '_'.join(scene_file.stem.split('_')[:2])
        
        scene_dir = base_path / scene_id
        scene_dir.mkdir(exist_ok=True)
        
        for file in base_path.glob(f'{scene_id}_*'):
            if file.is_file():
                shutil.move(str(file), str(scene_dir / file.name))
        
        txt_file = scene_dir / f'{scene_id}.txt'
        if not txt_file.exists():
            with open(txt_file, 'w') as f:
                f.write('axisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n')

if __name__ == '__main__':
    fix_scannet_structure() 