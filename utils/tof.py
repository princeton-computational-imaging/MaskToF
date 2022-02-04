import torch
import numpy as np
from scipy.constants import speed_of_light
from itertools import product, combinations


def sim_quad(depth, f, T, g, e): # convert
    """Simulate quad amplitude for 3D time-of-flight cameras
    Args:
        depth (tensor [B,1,H,W]): [depth map]
        f (scalar): [frequency in Hz]
        T (scalar): [integration time. metric not defined]
        g (scalar): [gain of the sensor. metric not defined]
        e (tensor [B,1,H,W]): [number of electrons created by a photon incident to the sensor. metric not defined]

    Returns:
        amplitudes (tensor [B,4,H,W]): [Signal amplitudes at 4 phase offsets.]
    """
    
    tru_phi = depth2phase(depth, f)
    
    A0 = g*e*(0.5+torch.cos(tru_phi)/np.pi)*T
    A1 = g*e*(0.5-torch.sin(tru_phi)/np.pi)*T
    A2 = g*e*(0.5-torch.cos(tru_phi)/np.pi)*T
    A3 = g*e*(0.5+torch.sin(tru_phi)/np.pi)*T
    
    return torch.stack([A0, A1, A2, A3], dim=1)


def decode_quad(amplitudes, T, mT): # convert
    """Simulate solid-state time-of-flight range camera.
    Args:
        amplitudes (tensor [B,4,H,W]): [Signal amplitudes at 4 phase offsets. See sim_quad().]
        T (scalar): [Integration time. Metric not defined.]
        mT (scalar): [Modulation period]

    Returns:
        phi_est, amplitude_est, offset_est (tuple(tensor [B,1,H,W])): [Estimated phi, amplitude, and offset]
    """
    assert amplitudes.shape[1] % 4 == 0
    
    A0, A1, A2, A3 = amplitudes[:,0::4,...], amplitudes[:,1::4,...], amplitudes[:,2::4,...], amplitudes[:,3::4,...]
    sigma = np.pi * T / mT
    
    phi_est = torch.atan2((A3-A1),(A0-A2))
    phi_est[phi_est<0] = phi_est[phi_est<0] + 2*np.pi
    
    amplitude_est = (sigma/T*np.sin(sigma)) * (( (A3-A1)**2 + (A0-A2)**2 )**0.5)/2 
    offset_est = (A0+A1+A2+A3)/(4*T)

    return phi_est, amplitude_est, offset_est

def depth2phase(depth, freq): # convert
    """Convert depth map to phase map.
    Args:
        depth (tensor [B,1,H,W]): Depth map (mm)
        freq (scalar): Frequency (hz)

    Returns:
        phase (tensor [B,1,H,W]): Phase map (radian)
    """

    tru_phi = (4*np.pi*depth*freq)/(1000*speed_of_light)
    return tru_phi

def phase2depth(phase, freq): # convert
    """Convert phase map to depth map.
    Args:
        phase (tensor [B,1,H,W]): Phase map (radian)
        freq ([type]): Frequency (Hz)

    Returns:
        depth (tensor [B,1,H,W]): Depth map (mm)
    """

    depth = (1000*speed_of_light*phase)/(4*np.pi*freq)
    return depth

def unwrap_ranking(phiList, f_list, min_depth=5000, max_depth=10000):

    """Efficient Multi-Frequency Phase Unwrapping using Kernel Density Estimation
    Args:
        phiList (list(K x tensor [B,4,H,W])): K different wrapped phases measured at the given frequencies f_list
        f_list (list [K]): K different frequencies in Hz.
        min_depth (scalar) : min depth in mm
        max_depth (scalar) : max depth in mm
    Returns:
        depth (tensor [B,1,H,W]): Unwrapped depth map (mm)
    """
    
    B,H,W = phiList[0].shape
    device = phiList[0].device
    
    # Compute the range of potential n (wraps)
    N = len(f_list)
    min_n = torch.zeros((N),dtype=np.int)
    max_n = torch.zeros((N),dtype=np.int)
    for i in range(N):
        min_phase = depth2phase(min_depth, f_list[i])
        max_phase = depth2phase(max_depth, f_list[i])
        min_n[i] = np.floor(min_phase/(2*np.pi))
        max_n[i] = np.floor(max_phase/(2*np.pi))
        
    n_list = []
    for i in range(N):
        n_list_ = []
        for n in range(min_n[i], max_n[i]+1, 1):
            n_list_.append(n)
        n_list.append(n_list_)

    from itertools import product, combinations
    prod = list(product(*n_list))
    M = len(prod) # number of potential combinations

    k = np.lcm.reduce(np.array(f_list).astype(np.int))/f_list
    
    t = []
    for i in range(N):
        t.append( (phiList[i]/(2*np.pi))*k[i] )
    t = torch.stack(t) # [N, B, H, W]
    
    err_list = torch.zeros((M,B,H,W), device=device)
    for i in range(M):
        n = prod[i]

        pairs = list(combinations(n, 2))
        kpairs = list(combinations((np.linspace(0,N-1,N)).astype(np.int), 2))
        for j, pair in enumerate(pairs):
            kpair = kpairs[j]
            k1 = k[kpair[0]]
            k2 = k[kpair[1]]
            n1 = pair[0]
            n2 = pair[1]

            err = (k1*n1 - k2*n2 - (t[kpair[1]] - t[kpair[0]]))**2
            w = min(1/(((k1/(2*np.pi))**2 + (k2/(2*np.pi))**2)*0.16871118634340782), 10.0)
            err_list[i,:,:,:] += err * w
            
    # sort and select the top three
    ind_sorted = torch.argsort(err_list, axis=0)
    ind_min = ind_sorted[0]
    ind_second = ind_sorted[1]
    ind_third = ind_sorted[2]
    
    # get the best one
    n_list_best = torch.zeros((N,B,H,W), device=device)

    for i in range(M):
        m = (ind_min == i)
        for j in range(N):
            n_list_best_i = torch.zeros((B,H,W), device=device)
            n_list_best_i[m] = prod[i][j]
            n_list_best[j,...] += n_list_best_i

    # simple phase unwrapping with the ranking
    up = []
    for i in range(N):
        up_i = phiList[i]+2*np.pi*n_list_best[i,...]
        up.append(up_i)
    up = np.array(up)    

    depth = []
    for i in range(N):
        depth_i = phase2depth(up[i], f_list[i])
        depth.append(depth_i)

    depth = torch.stack(depth)
    
    depth = depth.mean(axis=0)
    return depth
