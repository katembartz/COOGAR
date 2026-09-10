import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datetime import datetime
import shutil

import numpy as np
import nibabel as nib

from .coogar import get_capacities
from .coogar import get_mincut
from .coogar import save_progress

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input image to unwrap")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Temporary directory to store intermediate results")
    parser.add_argument("--outerIter", type=int, default=10, help="Maximum number of cuts applied")
    parser.add_argument("--p", type=int, default=2, help="Regularizer")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID if GPU is to be used")
    parser.add_argument("--padding", type=int, default=20, help="Padding to reduce border artifacts")
    parsed = parser.parse_args(args)
    
    parent_dir = parsed.input.parent
        
    if parsed.tmp_dir == None:
        now = datetime.now()
        formatted = now.strftime("%Y_%m_%d_%H_%M_%S")
        res_dir = Path(f"{str(parent_dir)}/coogar_tmp_{formatted}")
        parsed.tmp_dir = res_dir
        res_dir.mkdir(exist_ok=True)
        print(f'All temporary results will be saved to {parsed.tmp_dir}')
   
    img = nib.load(parsed.input)
    img_header = img.header
    img_affine = img.affine
    img_name = Path(Path(parsed.input).stem).stem
    g = img.get_fdata()
    g = torch.from_numpy(g).to(torch.float32)
    p = int(parsed.p)
    
    pd = parsed.padding
    
    if pd > 0:
        g_padded = F.pad(
            g.unsqueeze(0).unsqueeze(0), 
            pad=(pd, pd, pd, pd, pd, pd),      
            mode='replicate'
        )
        g = g_padded.squeeze(0).squeeze(0)
        
    u = g
        
    device = torch.device(f'cuda:{parsed.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    jumpSchedule = 2 * np.pi * (2 * (np.arange(1, parsed.outerIter + 1) % 2) - 1)
    jumpSchedule = torch.from_numpy(jumpSchedule)
    track_energy = []
    
    delta = torch.tensor([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
    delta = delta.to(device)
    
    Nx, Ny, Nz = g.shape
    
    # modify parameters for mincut algorithm
    params = {'maxIter': 5000,
                'tol': 0.001,
                'Nx': Nx,
                'Ny': Ny,
                'Nz': Nz,
                'delta': delta,
                'savecost': 1}

    
    for i in range(parsed.outerIter):
        print(f"Determining cut {i+1}")
        u = u.to(device)
        g = g.to(device)
        
        alpha = jumpSchedule[i].to(device)
        
        graph = get_capacities(u, g, delta, alpha, p, device)
        [cut, J] = get_mincut(graph, params, device)
        
        temp = u
        u = u + alpha*cut
        
        u_save = u[pd:-pd, pd:-pd, pd:-pd]
        
        # compute energy after cut (goal: minimize)
        graph_update = get_capacities(u, g, delta, alpha, p, device)
        Γ = graph_update['Γ'].detach().cpu().numpy()
        en = np.sum(Γ)
        print(en)
        track_energy.append(en)
        
        save_progress(temp, u_save, cut, parsed.tmp_dir, i, J, img_name, img_affine)
        
        if en < np.pi:
            shutil.copy(f'{parsed.tmp_dir}/{img_name}_gc_{i}.nii.gz', f'{parent_dir}/{img_name}_coogar-unw.nii.gz')
            print(f'Unwrapped result saved in {parent_dir}/{img_name}_coogar-unw.nii.gz')
            
    index = track_energy.index(min(track_energy))
    shutil.copy(f'{parsed.tmp_dir}/{img_name}_gc_{index}.nii.gz', f'{parent_dir}/{img_name}_coogar-unw.nii.gz')
    print(f'Unwrapped result saved in {parent_dir}/{img_name}_coogar-unw.nii.gz')