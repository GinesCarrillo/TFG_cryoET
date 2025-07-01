import numpy as np

def is_surrounded(matrix, row, col):
    """Verifica si el pixel en (row, col) está rodeado por valores distintos de cero."""
    rows, cols = matrix.shape
    eps = 1 # Umbral para considerar un valor como cero
    # Comprobamos las 8 posiciones alrededor de (row, col)
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            if (0 <= r < rows and 0 <= c < cols) and (r != row or c != col):
                if abs(matrix[r, c]) < eps:
                    return False
    return True

def reduce_matrix_hard(matrix, eps=1):
    # Convertimos la matriz a un array de NumPy para un mejor manejo
    np_matrix = np.array(matrix)
    # np.set_printoptions(precision=1, suppress=True)
    # for row in matrix:
    #     print(" ".join(f"{val:.1f}" for val in row))
    # print("\n" * 10)
    # Encontramos las posiciones de los valores distintos de cero
    non_zero_indices = np.argwhere(np.abs(np_matrix) > eps)
    # non_zero_indices = np.argwhere(np_matrix!=0)
    if non_zero_indices.size == 0:
        return np.zeros((0, 0))  # Si no hay elementos distintos de cero, devolvemos una matriz vacía

    # Obtenemos las filas y columnas mínimas y máximas
    min_row, min_col = non_zero_indices.min(axis=0)
    max_row, max_col = non_zero_indices.max(axis=0)

    # Creamos una lista para almacenar los valores válidos
    valid_values = []

    # Iteramos sobre los índices no cero
    for row, col in non_zero_indices:
        if not is_surrounded(np_matrix, row, col):
            valid_values.append((row, col))

    if not valid_values:
        return np_matrix  # Si no hay valores válidos, devolvemos la matriz original

    # Obtenemos las filas y columnas mínimas y máximas de los valores válidos
    valid_rows, valid_cols = zip(*valid_values)
    min_row = min(valid_rows)
    max_row = max(valid_rows)
    min_col = min(valid_cols)
    max_col = max(valid_cols)

    # Cortamos la matriz para mantener solo los valores válidos
    reduced_matrix = np_matrix[min_row:max_row + 1, min_col:max_col + 1]
    # for row in reduced_matrix:
    #     print(" ".join(f"{val:.1f}" for val in row))
    # print("\n" * 10)
    return reduced_matrix

def reduce_matrix(matrix, eps=1):
    # Convertimos la matriz a un array de NumPy para un mejor manejo
    np_matrix = np.array(matrix)
    # np.set_printoptions(precision=1, suppress=True)
    # for row in matrix:
    #     print(" ".join(f"{val:.1f}" for val in row))
    # print("\n" * 10)
    # Encontramos las posiciones de los valores distintos de cero
    non_zero_indices = np.argwhere(np.abs(np_matrix) > eps)
    # non_zero_indices = np.argwhere(np_matrix!=0)
    if non_zero_indices.size == 0:
        return np.zeros((0, 0))  # Si no hay elementos distintos de cero, devolvemos una matriz vacía

    # Obtenemos las filas y columnas mínimas y máximas
    min_row, min_col = non_zero_indices.min(axis=0)
    max_row, max_col = non_zero_indices.max(axis=0)
    
    #Tomamos el punto central
    central = (int((max_row + min_row) / 2) , int((max_col + min_col) / 2))
    # print(min_row, max_row)
    # print(min_col, max_col)
    
    #Buscamos el radio/diametro de la circunferencia que envuelve la figura
    radio = max(central[0]-min_row, max_row-central[0], central[1]-min_col, max_col-central[1])

    min_row = central[0]-radio
    max_row = central[0]+radio
    min_col = central[1]-radio
    max_col = central[1]+radio

    # Cortamos la matriz para mantener solo los valores válidos
    reduced_matrix = np_matrix[min_row:max_row + 1, min_col:max_col + 1]
    # for row in reduced_matrix:
    #     print(" ".join(f"{val:.1f}" for val in row))
    # print("\n" * 10)
    return reduced_matrix


