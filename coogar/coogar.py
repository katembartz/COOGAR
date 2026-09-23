import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# data fidelity cost component J
# negated in get_capacities
def cost1(u, g, device):
    return torch.cos(u - g).to(device)

# smoothness cost component V
def cost2(u, v, gamma, p, device):
    return (gamma * (u - v)**p).to(device)

# linear operator D implemented as a function
def D(f, mask):
    Nx, Ny, Nz = f.shape
    out = torch.zeros(Nx, Ny, Nz, 6, dtype=f.dtype, device=f.device)
    
    # if mask is not included, the graph is built everywhere
    val_x = mask[:-1, :, :] & mask[1:, :, :]
    val_x = val_x.to(f.device)
    val_y = mask[:, :-1, :] & mask[:, 1:, :]
    val_y = val_y.to(f.device)
    val_z = mask[:, :, :-1] & mask[:, :, 1:]
    val_z = val_z.to(f.device)

    out[:-1, :, :, 0] = torch.where( # +x : f[i]-f[i+1]
        val_x,
        f[:-1, :, :] - f[1:, :, :],
        torch.zeros_like(f[:-1, :, :])
    ) 
    out[:, :-1, :, 1] = torch.where( # +y 
        val_y,
        f[:, :-1, :] - f[:, 1:, :],
        torch.zeros_like(f[:, :-1, :])
    ) 
    out[:, :, :-1, 2] = torch.where( # +z
        val_z,
        f[:, :, :-1] - f[:, :, 1:],
        torch.zeros_like(f[:, :, :-1])
    )
    
    out[1:, :, :, 3] = torch.where( # +x : f[i]-f[i-1]
        val_x,
        f[1:, :, :] - f[:-1, :, :],
        torch.zeros_like(f[1:, :, :])
    ) 
    out[:, 1:, :, 4] = torch.where( # +y
        val_y,
        f[:, 1:, :] - f[:, :-1, :],
        torch.zeros_like(f[:, 1:, :])
    )
    out[:, :, 1:, 5] = torch.where( # +z
        val_z,
        f[:, :, 1:] - f[:, :, :-1],
        torch.zeros_like(f[:, :, 1:])
    )

    return out

# linear operator D_T implemented as a function
def D_T(f, mask):
    Nx, Ny, Nz, Nb = f.shape

    out = torch.zeros(
        Nx, Ny, Nz, Nb,
        dtype=f.dtype,
        device=f.device
    )

    val_x = (mask[:-1, :, :] & mask[1:, :, :]).to(f.device)
    val_y = (mask[:, :-1, :] & mask[:, 1:, :]).to(f.device)
    val_z = (mask[:, :, :-1] & mask[:, :, 1:]).to(f.device)

    out[1:, :, :, 0] = torch.where(
        val_x,
        f[1:, :, :, 0] - f[:-1, :, :, 0],
        torch.zeros_like(f[1:, :, :, 0])
    )

    out[0, :, :, 0] += torch.where(
        mask[0, :, :],
        f[0, :, :, 0],
        torch.zeros_like(f[0, :, :, 0])
    )

    out[-1, :, :, 0] -= torch.where(
        mask[-1, :, :],
        f[-1, :, :, 0],
        torch.zeros_like(f[-1, :, :, 0])
    )

    out[:, 1:, :, 1] = torch.where(
        val_y,
        f[:, 1:, :, 1] - f[:, :-1, :, 1],
        torch.zeros_like(f[:, 1:, :, 1])
    )

    out[:, 0, :, 1] += torch.where(
        mask[:, 0, :],
        f[:, 0, :, 1],
        torch.zeros_like(f[:, 0, :, 1])
    )

    out[:, -1, :, 1] -= torch.where(
        mask[:, -1, :],
        f[:, -1, :, 1],
        torch.zeros_like(f[:, -1, :, 1])
    )

    out[:, :, 1:, 2] = torch.where(
        val_z,
        f[:, :, 1:, 2] - f[:, :, :-1, 2],
        torch.zeros_like(f[:, :, 1:, 2])
    )

    out[:, :, 0, 2] += torch.where(
        mask[:, :, 0],
        f[:, :, 0, 2],
        torch.zeros_like(f[:, :, 0, 2])
    )

    out[:, :, -1, 2] -= torch.where(
        mask[:, :, -1],
        f[:, :, -1, 2],
        torch.zeros_like(f[:, :, -1, 2])
    )

    out[:-1, :, :, 3] = torch.where(
        val_x,
        f[:-1, :, :, 3] - f[1:, :, :, 3],
        torch.zeros_like(f[:-1, :, :, 3])
    )

    out[-1, :, :, 3] += torch.where(
        mask[-1, :, :],
        f[-1, :, :, 3],
        torch.zeros_like(f[-1, :, :, 3])
    )

    out[0, :, :, 3] -= torch.where(
        mask[0, :, :],
        f[0, :, :, 3],
        torch.zeros_like(f[0, :, :, 3])
    )

    out[:, :-1, :, 4] = torch.where(
        val_y,
        f[:, :-1, :, 4] - f[:, 1:, :, 4],
        torch.zeros_like(f[:, :-1, :, 4])
    )

    out[:, -1, :, 4] += torch.where(
        mask[:, -1, :],
        f[:, -1, :, 4],
        torch.zeros_like(f[:, -1, :, 4])
    )

    out[:, 0, :, 4] -= torch.where(
        mask[:, 0, :],
        f[:, 0, :, 4],
        torch.zeros_like(f[:, 0, :, 4])
    )

    out[:, :, :-1, 5] = torch.where(
        val_z,
        f[:, :, :-1, 5] - f[:, :, 1:, 5],
        torch.zeros_like(f[:, :, :-1, 5])
    )

    out[:, :, -1, 5] += torch.where(
        mask[:, :, -1],
        f[:, :, -1, 5],
        torch.zeros_like(f[:, :, -1, 5])
    )

    out[:, :, 0, 5] -= torch.where(
        mask[:, :, 0],
        f[:, :, 0, 5],
        torch.zeros_like(f[:, :, 0, 5])
    )

    return out

"""
    The key equation is:

    y_next = prox(y - τ * (1/2)*ΓD(ΓD)^Ty + ΓDw)
           = prox(y - τ * (1/2)*ΓDz + ΓDw) where z = (ΓD)^Ty, and we have -Γ = Γ^T
           = prox(y - τ * ΓD[(1/2)*z + w)] by factoring out ΓD
           = prox(y - τ * ΓDγ) where γ = (1/2)*z + w
"""
def get_mincut(graph, mask, params):
    # get graph components for cut by continuous optimization
    w = graph['w']
    Γ = graph['Γ']
    
    # access parameter information
    Nx = params['Nx']
    Ny = params['Ny']
    Nz = params['Nz']
    tol = params['tol']
    maxIter = params['maxIter']
    delta = params['delta']
    J = {}
    if params['savecost'] == 1:
        J['p'] = torch.zeros((maxIter, 1), dtype=w.dtype, device=w.device)
        J['d'] = torch.zeros((maxIter, 1), dtype=w.dtype, device=w.device)
        J['g'] = torch.zeros((maxIter, 1), dtype=w.dtype, device=w.device)
    Nb, _ = delta.shape
    
    maxVal = Γ.abs().max().item()
    
    τ = (1/(2*Nb))/ maxVal**2
    y = torch.zeros((Nx, Ny, Nz, Nb), dtype=w.dtype, device=w.device)
    
    Jg = 1
    
    # dual gradient descent update
    pbar = tqdm(range(maxIter), total=maxIter)
    for it in pbar:
        z = D_T(Γ * y, mask).sum(-1)   
        γ = 0.5 * z + w
        y = y - τ * (Γ * D(γ, mask))

        # proximal mapping
        y = torch.clamp(y, min=-1.0, max=1.0)
        
        # get and save costs (with new y -- need update)
        z = D_T(Γ * y, mask).sum(-1)
        γ = 0.5 * z + w

        ϕ = -γ        
        ϕ_cost = Γ * D(ϕ, mask) # Note: Γ = Γ^T

        Jp = (ϕ + w).pow(2).sum() + ϕ_cost.abs().sum()
        Jd = -((z * w).sum() + 0.25 * z.pow(2).sum())
        Jg = 2*(Jp-Jd)/(Jp+Jd)
        
        if params['savecost'] == 1:            
            J['p'][it] = Jp
            J['d'][it] = Jd
            J['g'][it] = Jg
        
        pbar.set_postfix(Jg=Jg.item())
        
        if Jg <= tol:
            break
        
    # generate cut
    cut = ϕ >= 0 
    return cut, J

def get_capacities(u, g, mask, delta, alpha, p, device):
    Nx, Ny, Nz = u.shape
    Nb = len(delta)
    w = torch.zeros((Nx, Ny, Nz), dtype=torch.float64, device=device)
    Γ = torch.zeros((Nx, Ny, Nz, Nb), dtype=torch.float64, device=device)
    
    # if mask is not included, the graph is built everywhere
    val_x = mask[:-1, :, :] & mask[1:, :, :]
    val_x = val_x.to(device)
    val_y = mask[:, :-1, :] & mask[:, 1:, :]
    val_y = val_y.to(device)
    val_z = mask[:, :, :-1] & mask[:, :, 1:]
    val_z = val_z.to(device)
    
    # add data edges
    # ==============
    p_new = u + alpha
    J0 = cost1(u, g, device)
    J1 = cost1(p_new, g, device)
    w = J1 - J0
    
    # add interaction edges
    # =====================
    gamma_x = val_x * torch.ones((Nx-1, Ny, Nz), device=device)
    gamma_y = val_y * torch.ones((Nx, Ny-1, Nz), device=device)
    gamma_z = val_z * torch.ones((Nx, Ny, Nz-1), device=device)
    
    # forward x
    # ---------
    p_old = u[:Nx-1, :, :]
    p_new = p_old + alpha
    q_old = u[1:, :, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma_x, p, device)
    V01 = cost2(p_old, q_new, gamma_x, p, device)
    V10 = cost2(p_new, q_old, gamma_x, p, device)
    V11 = cost2(p_new, q_new, gamma_x, p, device)
    
    w[:Nx-1, :, :] = w[:Nx-1, :, :] + (V10-V00)
    w[1:, :, :] = w[1:, :, :] + (V11-V10)
    Γ[:Nx-1, :, :, 0] = Γ[:Nx-1, :, :, 0] + (V11-V01-V10+V00)
    
    # forward y
    # ---------
    p_old = u[:, :Ny-1, :]
    p_new = p_old + alpha
    q_old = u[:, 1:, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma_y, p, device)
    V01 = cost2(p_old, q_new, gamma_y, p, device)
    V10 = cost2(p_new, q_old, gamma_y, p, device)
    V11 = cost2(p_new, q_new, gamma_y, p, device)
    
    w[:, :Ny-1, :] = w[:, :Ny-1, :] + (V10-V00)
    w[:, 1:, :] = w[:, 1:, :] + (V11-V10)
    Γ[:, :Ny-1, :, 1] = Γ[:, :Ny-1, :, 1] + (V11-V01-V10+V00)
    
    # forward z
    # ---------
    p_old = u[:, :, :Nz-1]
    p_new = p_old + alpha
    q_old = u[:, :, 1:]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma_z, p, device)
    V01 = cost2(p_old, q_new, gamma_z, p, device)
    V10 = cost2(p_new, q_old, gamma_z, p, device)
    V11 = cost2(p_new, q_new, gamma_z, p, device)
    
    w[:, :, :Nz-1] = w[:, :, :Nz-1] + (V10-V00)
    w[:, :, 1:] = w[:, :, 1:] + (V11-V10)
    Γ[:, :, :Nz-1, 2] = Γ[:, :, :Nz-1, 2] + (V11-V01-V10+V00)
    
    # back x
    # ---------
    p_old = u[1:, :, :]
    p_new = p_old + alpha
    q_old = u[:Nx-1, :, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma_x, p, device)
    V01 = cost2(p_old, q_new, gamma_x, p, device)
    V10 = cost2(p_new, q_old, gamma_x, p, device)
    V11 = cost2(p_new, q_new, gamma_x, p, device)
    
    w[1:, :, :] = w[1:, :, :] + (V10-V00)
    w[:Nx-1, :, :] = w[:Nx-1, :, :] + (V11-V10)
    Γ[1:, :, :, 3] = Γ[1:, :, :, 3] +(V11-V01-V10+V00)
    
    # back y
    # ------
    p_old = u[:, 1:, :]
    p_new = p_old + alpha
    q_old = u[:, :Ny-1, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma_y, p, device)
    V01 = cost2(p_old, q_new, gamma_y, p, device)
    V10 = cost2(p_new, q_old, gamma_y, p, device)
    V11 = cost2(p_new, q_new, gamma_y, p, device)
    
    w[:, 1:, :] = w[:, 1:, :] + (V10-V00)
    w[:, :Ny-1, :] = w[:, :Ny-1, :] + (V11-V10)
    Γ[:, 1:, :, 4] = Γ[:, 1:, :, 4] + (V11-V01-V10+V00)
    
    # back z
    # ------
    p_old = u[:, :, 1:]
    p_new = p_old + alpha
    q_old = u[:, :, :Nz-1]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma_z, p, device)
    V01 = cost2(p_old, q_new, gamma_z, p, device)
    V10 = cost2(p_new, q_old, gamma_z, p, device)
    V11 = cost2(p_new, q_new, gamma_z, p, device)
    
    w[:, :, 1:] = w[:, :, 1:] + (V10-V00)
    w[:, :, :Nz-1] = w[:, :, :Nz-1] + (V11-V10)
    Γ[:, :, 1:, 5] = Γ[:, :, 1:, 5] + (V11-V01-V10+V00)
    
    graph = {'w': w,
             'Γ': -Γ} # Γ negated for mincut
    
    return graph

def save_progress(u_old_3D, u_new_3D, cut_3D, prog_dir, iter, J, img_name, img_affine):
    u_new_3D = u_new_3D.detach().cpu().numpy()
    
    # save the unwrapped image at each iteration
    img_save = nib.Nifti1Image(u_new_3D, img_affine)
    nib.save(img_save, f'{prog_dir}/{img_name}_gc_{iter}.nii.gz')
