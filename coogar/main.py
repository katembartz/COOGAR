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

def compute_en(u, g, mask, p):
    val_x = mask[:-1, :, :] & mask[1:, :, :]
    val_x = val_x.to(u.device)
    val_y = mask[:, :-1, :] & mask[:, 1:, :]
    val_y = val_y.to(u.device)
    val_z = mask[:, :, :-1] & mask[:, :, 1:]
    val_z = val_z.to(u.device)
    
    res_fx = torch.where( 
        val_x,
        (u[1:, :, :] - u[:-1, :, :])
        - (g[1:, :, :] - g[:-1, :, :]),
        torch.zeros_like(u[1:, :, :])
    ) 
    
    res_bx = torch.where( 
        val_x,
        (u[:-1, :, :] - u[1:, :, :])
        - (g[:-1, :, :] - g[1:, :, :]),
        torch.zeros_like(u[:-1, :, :])
    ) 
    
    res_fy = torch.where( 
        val_y,
        (u[:, 1:, :] - u[:, :-1, :])
        - (g[:, 1:, :] - g[:, :-1, :]),
        torch.zeros_like(u[:, 1:, :])
    ) 
        
    res_by = torch.where( 
        val_y,
        (u[:, :-1, :] - u[:, 1:, :])
        - (g[:, :-1, :] - g[:, 1:, :]),
        torch.zeros_like(u[:, :-1, :])
    )
    
    res_fz = torch.where( 
        val_z,
        (u[:, :, 1:] - u[:, :, :-1])
        - (g[:, :, 1:] - g[:, :, :-1]),
        torch.zeros_like(u[:, :, 1:])
    ) 
    
    res_bz = torch.where( 
        val_z,
        (u[:, :, :-1] - u[:, :, 1:])
        - (g[:, :, :-1] - g[:, :, 1:]),
        torch.zeros_like(u[:, :, :-1])
    )
    
    res_fx = res_fx.detach().cpu().numpy()
    en_fx = np.sum(np.abs(res_fx)**p)
    res_bx = res_bx.detach().cpu().numpy()
    en_bx = np.sum(np.abs(res_bx)**p)
    res_fy = res_fy.detach().cpu().numpy()
    en_fy = np.sum(np.abs(res_fy)**p)
    res_by = res_by.detach().cpu().numpy()
    en_by = np.sum(np.abs(res_by)**p)
    res_fz = res_fz.detach().cpu().numpy()
    en_fz = np.sum(np.abs(res_fz)**p)
    res_bz = res_bz.detach().cpu().numpy()
    en_bz = np.sum(np.abs(res_bz)**p)
    
    en = en_fx + en_bx + en_fy + en_by + en_fz + en_bz
    
    return en
    

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input image to unwrap")
    parser.add_argument("--out-dir", type=Path, default=None, help="Path to store unwrapped image")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Temporary directory to store intermediate results")
    parser.add_argument("--mask", type=Path, default=None, help="ROI (binary) mask if background is nulled. NOTE: this is important to include if the input image is masked.")
    parser.add_argument("--outerIter", type=int, default=16, help="Maximum number of cuts applied")
    parser.add_argument("--p", type=int, default=2, help="Regularizer")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID if GPU is to be used")
    parser.add_argument("--padding", type=int, default=0, help="Padding to reduce border artifacts")
    parsed = parser.parse_args(args)
    
    parent_dir = parsed.input.parent
    
    if parsed.out_dir == None:
        parsed.out_dir = parent_dir
        
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
    g = torch.from_numpy(g).to(torch.float64)
    p = int(parsed.p)
    
    Nx, Ny, Nz = g.shape
    
    if parsed.mask != None:
        mask_nib = nib.load(parsed.mask)
        mask_org = mask_nib.get_fdata()
    else:
        mask_org = np.ones((Nx, Ny, Nz), dtype=bool)
    mask_org = torch.from_numpy(mask_org).to(torch.bool)
    
    g = g * mask_org
    u_save = g # save image in original size
    
    # crop g around mask to save computational cost
    coords = torch.nonzero(mask_org)
    min_idx = coords.min(dim=0).values
    max_idx = coords.max(dim=0).values
    g_crop = g[min_idx[0] - 1 :max_idx[0] + 2, min_idx[1] - 1:max_idx[1] + 2, min_idx[2] - 1:max_idx[2] + 2]
    mask = mask_org[min_idx[0] - 1 :max_idx[0] + 2, min_idx[1] - 1:max_idx[1] + 2, min_idx[2] - 1:max_idx[2] + 2]
    
    # update shape sizes
    Nx, Ny, Nz = g_crop.shape
    
    pd = parsed.padding
    
    if pd > 0:
        g_padded = F.pad(
            g_crop.unsqueeze(0).unsqueeze(0), 
            pad=(pd, pd, pd, pd, pd, pd),      
            mode='replicate'
        )
        g_crop = g_padded.squeeze(0).squeeze(0)
        
    u = g_crop
        
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
    
    # modify parameters for mincut algorithm
    params = {'maxIter': 7500,
                'tol': 0.001,
                'Nx': Nx,
                'Ny': Ny,
                'Nz': Nz,
                'delta': delta,
                'savecost': 1}

    
    for i in range(parsed.outerIter):
        print(f"Determining cut {i+1}")
        u = u.to(device)
        g_crop = g_crop.to(device)
        mask = mask.to(device)
        
        alpha = jumpSchedule[i].to(device)
        
        graph = get_capacities(u, g_crop, mask, delta, alpha, p, device)       
        [cut, J] = get_mincut(graph, mask, params)
        
        temp = u
        u = u + alpha*cut
        
        if pd > 0:
            u_save[min_idx[0] - 1 :max_idx[0] + 2, min_idx[1] - 1:max_idx[1] + 2, min_idx[2] - 1:max_idx[2] + 2] = u[pd:-pd, pd:-pd, pd:-pd]
        else:
            u_save[min_idx[0] - 1 :max_idx[0] + 2, min_idx[1] - 1:max_idx[1] + 2, min_idx[2] - 1:max_idx[2] + 2] = u
        u_save = u_save * mask_org
        
        # compute energy after cut (goal: minimize)
        en = compute_en(u, g_crop, mask, p)  
        print(en)
        track_energy.append(en)
        
        save_progress(temp, u_save, cut, parsed.tmp_dir, i, J, img_name, img_affine)
            
    index = track_energy.index(min(track_energy))
    shutil.copy(f'{parsed.tmp_dir}/{img_name}_gc_{index}.nii.gz', f'{parsed.out_dir}/{img_name}_coogar-unw.nii.gz')
    print(f'Unwrapped result saved in {parsed.out_dir}/{img_name}_coogar-unw.nii.gz')