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
