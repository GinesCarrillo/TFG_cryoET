import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_function_on_circle(f_theta, num_points=360):
    """
    Representa una función f(θ) en un espacio 3D sobre una circunferencia.

    Parámetros:
    - f: Función a graficar, que toma θ en grados y devuelve un valor f(θ).
    - num_points: Número de puntos para la representación.
    """
    # 1. Generar valores de θ en grados y convertirlos a radianes
    theta_deg = np.linspace(0, 360, num_points)
    theta_rad = np.radians(theta_deg)  # Conversión a radianes



    # 3. Convertir a coordenadas cartesianas
    x = np.cos(theta_rad)  # Posición en el eje X
    y = np.sin(theta_rad)  # Posición en el eje Y
    z = f_theta  # La función se representa en el eje Z

    # 4. Graficar en 3D
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, label="f(θ)", color='b')
    ax.set_xlabel("X (cos θ)")
    ax.set_ylabel("Y (sin θ)")
    ax.set_zlabel("f(θ)")
    ax.set_title("Representación 3D de f(θ) sobre la Circunferencia")
    ax.legend()
    
    # Mostrar gráfico
    plt.show()
    
