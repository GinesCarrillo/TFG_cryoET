import torch

def weighted_euler_curve(V, E, F, V_weights, E_weights, F_weights, fun, stepsize=100):
    """
    Calcula la curva de característica de Euler ponderada para una función definida en los vértices.

    Parámetros:
    - V: [N, 2] tensor de vértices
    - E: [2, M] tensor de aristas
    - F: [3, K] tensor de caras
    - V_weights: [N, 1] pesos de vértices
    - E_weights: [M, 1] pesos de aristas
    - F_weights: [K, 1] pesos de caras
    - fun: [N] valores de la función escalar en los vértices (por ejemplo, proyección)
    - stepsize: int, número de niveles de filtración

    Retorna:
    - curve: [stepsize+1, 2] tensor con niveles y valores de característica de Euler
    """
    if V.shape[0] != fun.shape[0]:
        raise ValueError("fun debe tener la misma longitud que el número de vértices.")

    # 1. Obtener valores de la función para aristas y caras (máximo de sus vértices)
    fe = torch.max(fun[E], dim=0).values  # [M]
    ff = torch.max(fun[F], dim=0).values  # [K]

    # 2. Crear los umbrales de filtración
    thresholds = torch.linspace(-1, 1, stepsize + 1)  # [stepsize + 1]
    curve = torch.zeros(stepsize + 1, 2)

    curve[:, 0] = thresholds  # primera columna = umbrales

    # 3. Calcular la curva de EC ponderada para cada nivel
    for i, t in enumerate(thresholds):
        v_mask = fun <= t
        e_mask = fe <= t
        f_mask = ff <= t

        v_sum = V_weights[v_mask].sum()
        e_sum = E_weights[e_mask].sum()
        f_sum = F_weights[f_mask].sum()

        curve[i, 1] = v_sum - e_sum + f_sum

    return curve
