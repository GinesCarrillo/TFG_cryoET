import torch
import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from weighted_euler_curve import weighted_euler_curve  # asegúrate de tenerla implementada
import math
def complex_to_weighted_ECT(complex, num_directions, num_steps,
                            method='none', window=5, normalization_method='none'):
    """
    Calcula la SWECT (Smoothed Weighted Euler Characteristic Transform).

    Entradas:
    - complex: dict con claves V, E, F, V_weights, E_weights, F_weights
    - num_directions: número de direcciones sobre el círculo unitario
    - num_steps: número de pasos de la curva EC
    - method: método de suavizado ('none', 'movmean', 'gaussian')
    - window: tamaño de la ventana de suavizado
    - normalization_method: 'none', 'max', 'total', 'ECT'

    Retorna:
    - SWECT: tensor [num_steps, num_directions]
    """

    V = complex['V']
    E = complex['E']
    F = complex['F']
    V_weights = complex['V_weights']
    E_weights = complex['E_weights']
    F_weights = complex['F_weights']

    # Normalización de pesos
    if normalization_method == 'max':
        norm_factor = V_weights.max()
        V_weights = V_weights / norm_factor
        E_weights = E_weights / norm_factor
        F_weights = F_weights / norm_factor
    elif normalization_method == 'total':
        V_weights = V_weights / V_weights.sum()
        E_weights = E_weights / E_weights.sum()
        F_weights = F_weights / F_weights.sum()
    elif normalization_method == 'ECT':
        V_weights = torch.ones_like(V_weights)
        E_weights = torch.ones_like(E_weights)
        F_weights = torch.ones_like(F_weights)
    # Si es 'none', no se hace nada

    # Centrado y normalización espacial
    Z = V - V.mean(dim=0, keepdim=True)
    max_radius = torch.norm(Z, dim=1).max()
    Z = Z / max_radius

    # Generar direcciones en S^1
    theta = torch.linspace(-np.pi, np.pi, steps=num_directions + 1)[:-1]  # excluye el último para evitar duplicado
    directions = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # [num_directions, 2]

    SWECT = torch.zeros(num_steps, num_directions)

    for i, d in enumerate(directions):
        fun = Z @ d  # proyección en la dirección d
        EC = weighted_euler_curve(Z, E, F, V_weights, E_weights, F_weights, fun, num_steps)
        y = EC[:num_steps, 1]  # solo la EC

        # Suavizado opcional
        if method == 'gaussian':
            y_smoothed = torch.tensor(gaussian_filter1d(y.numpy(), sigma=window))
        elif method in ['movmean', 'mean', 'uniform']:
            y_smoothed = torch.tensor(uniform_filter1d(y.numpy(), size=window))
        else:
            y_smoothed = y

        SWECT[:, i] = y_smoothed

    return SWECT


import torch

def distance_rotation_invariant(WECT1: torch.Tensor, WECT2: torch.Tensor):
    """
    Calcula la distancia L2 mínima entre WECT1 y todas las rotaciones cíclicas de WECT2.

    Parámetros:
    - WECT1: [num_steps, num_directions] tensor
    - WECT2: [num_steps, num_directions] tensor

    Retorna:
    - dist: distancia L2 mínima
    - shift: desplazamiento cíclico (en columnas) que minimiza la distancia
    """

    assert WECT1.shape == WECT2.shape, "Ambos WECTs deben tener la misma forma"
    num_directions = 360

    min_dist = float('inf')
    best_shift = 0
    angles=[]
    distancias=[]
    for d in range(num_directions):
        WECT2_shifted = torch.roll(WECT2, shifts=-d, dims=1)
        dist = torch.norm(WECT1 - WECT2_shifted, p='fro')
        angles.append(math.degrees((2 * math.pi * d) / num_directions))
        distancias.append(dist)
        
        if dist < min_dist:
            min_dist = dist
            best_shift = d
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(angles, [d.item() for d in distancias], marker='o')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('L2 Distance')
    plt.title('L2 Distance vs Angle Shift')
    plt.grid(True)
    plt.show()
    angle = math.degrees((2 * math.pi * best_shift) / num_directions)  # grados
    return min_dist.item(), best_shift, angle
