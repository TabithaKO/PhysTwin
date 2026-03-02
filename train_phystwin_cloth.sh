#!/bin/bash
# Train all PhysTwin cloth cases sequentially
cd ~/PhysTwin

python -c "
import glob, os, json

base_path = './data/different_types'
cloth_cases = sorted([d.split('/')[-1] for d in glob.glob(f'{base_path}/*') if 'cloth' in d.split('/')[-1]])

print(f'Training {len(cloth_cases)} cloth cases')
for i, case in enumerate(cloth_cases):
    with open(f'{base_path}/{case}/split.json') as f:
        split = json.load(f)
    train_frame = split['train'][1]
    print(f'\n[{i+1}/{len(cloth_cases)}] {case} (train_frame={train_frame})')
    ret = os.system(f'python train_warp.py --base_path {base_path} --case_name {case} --train_frame {train_frame}')
    if ret != 0:
        print(f'FAILED: {case}')
    else:
        print(f'Done: {case}')
"
