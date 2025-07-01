import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize
from scipy.spatial import Delaunay
from collections import defaultdict

def build_weighted_complex(image_array, threshold = 0.01):
    """
    Equivalente Python del script de MATLAB build_weighted_complex.
    Entrada: imagen en escala de grises como array numpy normalizado [0, 1]
    Salida: V, E, F, V_weights, E_weights, F_weights (todos como tensores)
    """

    image = image_array
    height, width = image.shape

    # 1. Encontrar píxeles activos (valor > 0)
    row, col = np.nonzero(image)
    # Filtrar valores muy próximos a cero
    valid_indices = image[row, col] > threshold
    row, col = row[valid_indices], col[valid_indices]

    V_centers = np.stack([col, height - 1 - row], axis=1)  # Coordenadas euclídeas
    num_centers = V_centers.shape[0]

    # 2. Pesos de los centros
    V_center_weights = np.array([image[r, c] for r, c in zip(row, col)], dtype=np.float32)

    # 3. Crear vértices de esquinas
    V_corners = []
    for cx, cy in V_centers:
        V_corners.extend([
            (cx - 0.5, cy - 0.5),
            (cx + 0.5, cy - 0.5),
            (cx - 0.5, cy + 0.5),
            (cx + 0.5, cy + 0.5),
        ])
    V_corners = np.unique(V_corners, axis=0)

    # 4. Unir todos los vértices
    V_all = np.vstack([V_centers, V_corners])
    V = torch.tensor(V_all, dtype=torch.float32)

    # 5. Crear 4 caras por centro
    F = []
    for i, (cx, cy) in enumerate(V_centers):
        center_idx = i
        offsets = np.array([
            [+0.5, +0.5],
            [-0.5, +0.5],
            [-0.5, -0.5],
            [+0.5, -0.5]
        ])
        neighbors = V_all.tolist()
        corner_indices = []
        for off in offsets:
            corner = (cx + off[0], cy + off[1])
            idx = np.where((V_all == corner).all(axis=1))[0][0]
            corner_indices.append(idx)

        F.append([center_idx, corner_indices[0], corner_indices[1]])
        F.append([center_idx, corner_indices[1], corner_indices[2]])
        F.append([center_idx, corner_indices[2], corner_indices[3]])
        F.append([center_idx, corner_indices[3], corner_indices[0]])

    F = torch.tensor(F, dtype=torch.long)

    # 6. Extraer aristas únicas de las caras
    edge_set = set()
    for face in F.tolist():
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            u, v = sorted((face[a], face[b]))
            edge_set.add((u, v))
    E = torch.tensor(sorted(edge_set), dtype=torch.long)

    # 7. Pesos de caras (igual a los pesos del centro del píxel)
    F_weights = torch.tensor(np.repeat(V_center_weights, 4), dtype=torch.float32)

    # 8. Pesos de vértices
    V_weights = torch.zeros(V.shape[0], dtype=torch.float32)
    V_weights[:num_centers] = torch.tensor(V_center_weights)

    for vi in range(num_centers, V.shape[0]):
        face_rows = (F == vi).any(dim=1).nonzero(as_tuple=True)[0]
        if len(face_rows) > 0:
            V_weights[vi] = F_weights[face_rows].max()

    # 9. Pesos de aristas
    E_weights = torch.zeros(E.shape[0], dtype=torch.float32)
    for i, (u, v) in enumerate(E.tolist()):
        face_rows = ((F == u) | (F == v)).sum(dim=1) == 2
        if face_rows.any():
            E_weights[i] = F_weights[face_rows].max()

    return V, E.T, F.T, V_weights.unsqueeze(1), E_weights.unsqueeze(1), F_weights.unsqueeze(1)

import matplotlib.pyplot as plt

def plot_complex(complex, show_faces=True, show_edges=True, show_vertices=True,
                 figsize=(6, 6), title="Complejo simplicial"):
    V = complex['V'].numpy()
    E = complex['E'].numpy()
    F = complex['F'].numpy()
    Vw = complex['V_weights'].squeeze().numpy()
    
    plt.figure(figsize=figsize)
    plt.title(title)

    # Mostrar caras (triángulos)
    if show_faces:
        for tri in F.T:
            pts = V[tri]
            plt.fill(pts[:, 0], pts[:, 1], color='lightgray', alpha=0.3, edgecolor='gray', linewidth=0.5)

    # Mostrar aristas
    if show_edges:
        for u, v in E.T:
            x1, y1 = V[u]
            x2, y2 = V[v]
            plt.plot([x1, x2], [y1, y2], color='gray', linewidth=0.7)

    # Mostrar vértices coloreados por su peso
    if show_vertices:
        sc = plt.scatter(V[:, 0], V[:, 1], c=Vw, cmap='plasma', s=10, zorder=2)
        plt.colorbar(sc, label='Peso de vértice (intensidad)')

    plt.axis("equal")
    plt.axis("off")
    plt.show()
