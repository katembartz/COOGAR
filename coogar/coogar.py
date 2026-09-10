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
def D(f, dx, dy, dz, Nx, Ny, Nz, device): 
    v = f.clone()
    c = f.clone()
    
    start_x = max(0, -dx)
    stop_x = min(Nx-dx-1, Nx-1)
    start_y = max(0, -dy)
    stop_y = min(Ny-dy-1, Ny-1)
    start_z = max(0, -dz)
    stop_z = min(Nz-dz-1, Nz-1)
    
    v[start_x:stop_x, start_y:stop_y, start_z:stop_z] = f[start_x:stop_x, start_y:stop_y, start_z:stop_z] - f[start_x+dx:stop_x+dx, start_y+dy:stop_y+dy, start_z+dz:stop_z+dz]
    c[start_x:stop_x, start_y:stop_y, start_z:stop_z] = torch.zeros((Nx-abs(dx)-1, Ny-abs(dy)-1, Nz-abs(dz)-1), dtype=torch.float32, device=device)
    v = v - c
    
    return v

# linear operator D_T implemented as a function
def D_T(f, dx, dy, dz, Nx, Ny, Nz, device):
    v = f.clone()
    c = f.clone()
    
    start_x = max(0, dx)
    stop_x = min(Nx+dx-1, Nx-1)
    start_y = max(0, dy)
    stop_y = min(Ny+dy-1, Ny-1)
    start_z = max(0, dz)
    stop_z = min(Nz+dz-1, Nz-1)
    
    v[start_x:stop_x, start_y:stop_y, start_z:stop_z] = f[start_x:stop_x, start_y:stop_y, start_z:stop_z] - f[start_x-dx:stop_x-dx, start_y-dy:stop_y-dy, start_z-dz:stop_z-dz]
    
    start_x = max(0, -dx)
    stop_x = min(Nx-dx-1, Nx-1)
    start_y = max(0, -dy)
    stop_y = min(Ny-dy-1, Ny-1)
    start_z = max(0, -dz)
    stop_z = min(Nz-dz-1, Nz-1)
    
    c[start_x:stop_x, start_y:stop_y, start_z:stop_z] = torch.zeros((Nx-abs(dx)-1, Ny-abs(dy)-1, Nz-abs(dz)-1), dtype=torch.float32, device=device)
    v = v - c
    
    return v

"""
    The key equation is:

    y_next = prox(y - τ * (1/2)*ΓD(ΓD)^Ty + ΓDw)
           = prox(y - τ * (1/2)*ΓDz + ΓDw) where z = (ΓD)^Ty, and we have -Γ = Γ^T
           = prox(y - τ * ΓD[(1/2)*z + w)] by factoring out ΓD
           = prox(y - τ * ΓDγ) where γ = (1/2)*z + w
"""
def get_mincut(graph, params, device):
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
        J['p'] = torch.zeros((maxIter, 1), dtype=torch.float32, device=device)
        J['d'] = torch.zeros((maxIter, 1), dtype=torch.float32, device=device)
        J['g'] = torch.zeros((maxIter, 1), dtype=torch.float32, device=device)
    Nb, _ = delta.shape
    
    maxVal = Γ.abs().max().item()
    
    τ = (1/(2*Nb))/ maxVal**2
    y = torch.zeros((Nx, Ny, Nz, Nb), dtype=torch.float32, device=device)
    
    Jg = 1
    iter = 0
    
    pbar = tqdm(range(maxIter), total=maxIter)
    for it in pbar:
        z = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=device)
        ρ = torch.zeros((Nx, Ny, Nz, Nb), dtype=torch.float32, device=device)
        
        for b in range(Nb):
            z = z + D_T(Γ[:, :, :, b] * y[:, :, :, b], delta[b, 0], delta[b, 1], delta[b,2], Nx, Ny, Nz, device)
        γ = 0.5*z + w
        
        for b in range(Nb):
            ρ[:, :, :, b] = Γ[:, :, :, b] * D(γ, delta[b, 0], delta[b, 1], delta[b,2], Nx, Ny, Nz, device)
            
        y = y - τ*ρ
        
        # proximal mapping
        y = torch.clamp(y, min=-1.0, max=1.0)
        
        ϕ_cost = torch.zeros((Nx, Ny, Nz, Nb), dtype=torch.float32, device=device)
        
        ϕ = -γ 
        
        for b in range(Nb):
            ϕ_cost[:, :, :, b] = Γ[:, :, :, b] * D(ϕ, delta[b, 0], delta[b, 1], delta[b, 2], Nx, Ny, Nz, device)
            
        w = w.float()
            
        Jp = (ϕ + w).pow(2).sum() + ϕ_cost.abs().sum()
        Jd = -((z * w).sum() + 0.25 * z.pow(2).sum())
        Jg = 2*(Jp-Jd)/(Jp+Jd)
        
        if params['savecost'] == 1:
            J['p'][it] = Jp
            J['d'][it] = Jd
            J['g'][it] = Jg
            
        pbar.set_postfix(Jg=Jg)
            
        if Jg <= tol:
            break
        
    # generate cut
    cut = ϕ > 0
    return cut, J

def get_capacities(u, g, delta, alpha, p, device):
    Nx, Ny, Nz = u.shape
    Nb = len(delta)
    w = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, device=device)
    Γ = torch.zeros((Nx, Ny, Nz, Nb), dtype=torch.float32, device=device)
    
    # add data edges
    # ==============
    p_new = u + alpha
    J0 = cost1(u, g, device)
    J1 = cost1(p_new, g, device)
    w = J1 - J0
    
    # add interaction edges
    # =====================
    gamma = 1 / (delta[:, 0]**2 + delta[:, 1]**2 + delta[:, 2]**2)
    
    # forward x
    # ---------
    p_old = u[:Nx-1, :, :]
    p_new = p_old + alpha
    q_old = u[1:, :, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma[0], p, device)
    V01 = cost2(p_old, q_new, gamma[0], p, device)
    V10 = cost2(p_new, q_old, gamma[0], p, device)
    V11 = cost2(p_new, q_new, gamma[0], p, device)
    
    w[:Nx-1, :, :] = w[:Nx-1, :, :] + (V10-V00)
    w[1:, :, :] = w[1:, :, :] + (V11-V10)
    Γ[:Nx-1, :, :, 0] = Γ[:Nx-1, :, :, 0] + (V11-V01-V10+V00)
    
    # forward y
    # ---------
    p_old = u[:, :Ny-1, :]
    p_new = p_old + alpha
    q_old = u[:, 1:, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma[1], p, device)
    V01 = cost2(p_old, q_new, gamma[1], p, device)
    V10 = cost2(p_new, q_old, gamma[1], p, device)
    V11 = cost2(p_new, q_new, gamma[1], p, device)
    
    w[:, :Ny-1, :] = w[:, :Ny-1, :] + (V10-V00)
    w[:, 1:, :] = w[:, 1:, :] + (V11-V10)
    Γ[:, :Ny-1, :, 1] = Γ[:, :Ny-1, :, 1] + (V11-V01-V10+V00)
    
    # forward z
    # ---------
    p_old = u[:, :, :Nz-1]
    p_new = p_old + alpha
    q_old = u[:, :, 1:]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma[2], p, device)
    V01 = cost2(p_old, q_new, gamma[2], p, device)
    V10 = cost2(p_new, q_old, gamma[2], p, device)
    V11 = cost2(p_new, q_new, gamma[2], p, device)
    
    w[:, :, :Nz-1] = w[:, :, :Nz-1] + (V10-V00)
    w[:, :, 1:] = w[:, :, 1:] + (V11-V10)
    Γ[:, :, :Nz-1, 2] = Γ[:, :, :Nz-1, 2] + (V11-V01-V10+V00)
    
    # back x
    # ---------
    p_old = u[1:, :, :]
    p_new = p_old + alpha
    q_old = u[:Nx-1, :, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma[3], p, device)
    V01 = cost2(p_old, q_new, gamma[3], p, device)
    V10 = cost2(p_new, q_old, gamma[3], p, device)
    V11 = cost2(p_new, q_new, gamma[3], p, device)
    
    w[1:, :, :] = w[1:, :, :] + (V10-V00)
    w[:Nx-1, :, :] = w[:Nx-1, :, :] + (V11-V10)
    Γ[1:, :, :, 3] = Γ[1:, :, :, 3] +(V11-V01-V10+V00)
    
    # back y
    # ------
    p_old = u[:, 1:, :]
    p_new = p_old + alpha
    q_old = u[:, :Ny-1, :]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma[4], p, device)
    V01 = cost2(p_old, q_new, gamma[4], p, device)
    V10 = cost2(p_new, q_old, gamma[4], p, device)
    V11 = cost2(p_new, q_new, gamma[4], p, device)
    
    w[:, 1:, :] = w[:, 1:, :] + (V10-V00)
    w[:, :Ny-1, :] = w[:, :Ny-1, :] + (V11-V10)
    Γ[:, 1:, :, 4] = Γ[:, 1:, :, 4] + (V11-V01-V10+V00)
    
    # back z
    # ------
    p_old = u[:, :, 1:]
    p_new = p_old + alpha
    q_old = u[:, :, :Nz-1]
    q_new = q_old + alpha
    
    V00 = cost2(p_old, q_old, gamma[5], p, device)
    V01 = cost2(p_old, q_new, gamma[5], p, device)
    V10 = cost2(p_new, q_old, gamma[5], p, device)
    V11 = cost2(p_new, q_new, gamma[5], p, device)
    
    w[:, :, 1:] = w[:, :, 1:] + (V10-V00)
    w[:, :, :Nz-1] = w[:, :, :Nz-1] + (V11-V10)
    Γ[:, :, 1:, 5] = Γ[:, :, 1:, 5] + (V11-V01-V10+V00)
    
    graph = {'w': w,
             'Γ': -Γ} # Γ negated for mincut
    
    return graph

def local_diff(f):
    Nx, Ny, Nz = f.shape
    out = torch.zeros(Nx, Ny, Nz, 6, dtype=f.dtype, device=f.device)

    out[:-1, :, :, 0] = f[:-1, :, :] - f[1:, :, :] # +x : f[i]-f[i+1]
    out[:, :-1, :, 1] = f[:, :-1, :] - f[:, 1:, :] # +y
    out[:, :, :-1, 2] = f[:, :, :-1] - f[:, :, 1:] # +z

    out[1:, :, :, 3] = f[1:, :, :] - f[:-1, :, :] # -x : f[i]-f[i-1]
    out[:, 1:, :, 4] = f[:, 1:, :] - f[:, :-1, :] # -y
    out[:, :, 1:, 5] = f[:, :, 1:] - f[:, :, :-1] # -z

    return out

def save_progress(u_old_3D, u_new_3D, cut_3D, prog_dir, iter, J, img_name, img_affine):
    # determine if (and where) wraps remain in the signal
    disc = local_diff(u_new_3D)
    disc = disc.detach().cpu().numpy()
    disc_3D = np.max(disc, axis=-1)
    wrp_bny = np.where(np.abs(disc_3D) >= 2*np.pi, disc_3D, 0)
    
    # compute energy in signal
    
    
    u_new_3D = u_new_3D.detach().cpu().numpy()
    u_old_3D = u_old_3D.detach().cpu().numpy()
    cut_3D = cut_3D.detach().cpu().numpy()
    
    # save the unwrapped image at each iteration
    img_save = nib.Nifti1Image(u_new_3D, img_affine)
    nib.save(img_save, f'{prog_dir}/{img_name}_gc_{iter}.nii.gz')
    
    # display the results for one slice and the costs for each iteration
    Jp = J['p'].detach().cpu()
    Jd = J['d'].detach().cpu()
    Jg = J['g'].detach().cpu()
    
    _, _, Nx = u_old_3D.shape
    disp = Nx // 3
    u_old = u_old_3D[:, :, disp]
    u_new = u_new_3D[:, :, disp]
    cut = cut_3D[:, :, disp]
    wrps = wrp_bny[:, :, disp]
    
    ln, _ = Jp.shape
    ln = range(ln)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), squeeze=False)
    
    axes[0, 0].imshow(u_old, cmap='gray')
    axes[0, 0].set_title(f'Input at iter: {iter}')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    axes[0, 1].imshow(cut, cmap='gray')
    axes[0, 1].set_title('Graph cut')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    axes[0, 2].imshow(u_new, cmap='gray')
    axes[0, 2].set_title('Signal with cut applied')
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    
    axes[1, 0].plot(ln, Jp, color='red', label = 'Primal Cost')
    axes[1, 0].plot(ln, Jd, color='blue', label = 'Dual Cost')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Cost')
    axes[1, 0].set_title('Primal and Dual Costs')
    axes[1, 0].legend()
    
    axes[1, 1].plot(ln, Jg, color='green')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Cost')
    axes[1, 1].set_title('Total Cost')
    
    axes[1, 1].plot(ln, Jg, color='green')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Cost')
    axes[1, 1].set_title('Total Cost')
    
    axes[1, 2].imshow(wrps, cmap='gray')
    axes[1, 2].set_title(f'Location of Remaining Wraps (white)')
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    plt.savefig(f'{prog_dir}/{iter}.png', dpi = 500, bbox_inches = 'tight')
