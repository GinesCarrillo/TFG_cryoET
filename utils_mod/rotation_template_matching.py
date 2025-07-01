import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import match_template
# from template import match_template as match_template2
# from template import match_template_weighted
from skimage.transform import rotate
from skimage.measure import label, regionprops
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from utils_mod.reduceMatrix import *
from utils_mod.matrices import *
import pandas as pd
from scipy.stats import pearsonr
from utils_mod.utils import *
import matplotlib.colors as mcolors
import cv2
from scipy.ndimage import uniform_filter
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colorbar import ColorbarBase

def mostrar_imagen(image):
    """
    Muestra una imagen dada una matriz 2D o 3D (con tercera dimensión de tamaño 1).
    
    Parameters:
    - image: Matriz de la imagen (2D o 3D con tercera dimensión de tamaño 1).
    """
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]  # Convertir a 2D si la tercera dimensión es de tamaño 1
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Ocultar los ejes
    plt.show()

def supresion_no_maxima(match_values, coords, template_shape, radius):
    """
    Aplica supresión no máxima para eliminar coincidencias cercanas que representen la misma región.
    
    Parameters:
    - match_values: Array de coincidencias por rotación.
    - coords: Coordenadas de los píxeles seleccionados.
    - template_shape: Tamaño de la plantilla (alto, ancho).
    
    Returns:
    - filtered_coords: Coordenadas de las mejores coincidencias después de la supresión no máxima.
    """
    # Ordenar las coordenadas en función del valor máximo de coincidencia en cada coordenada (de mayor a menor)
    max_per_pixel = np.max(match_values, axis=2)  # Obtener los valores máximos para cada pixel
    sorted_indices = np.argsort(max_per_pixel[coords[:, 0], coords[:, 1]])[::-1]  # Ordenar de mayor a menor
    sorted_coords = coords[sorted_indices]  # Coordenadas ordenadas por la mejor coincidencia
    
    # Inicializar lista de coordenadas filtradas
    filtered_coords = []
    
    # Tamaño de la ventana de supresión (basada en el tamaño de la plantilla)
    suppression_radius_y, suppression_radius_x = template_shape[0] // radius, template_shape[1] // radius
    
    # Marcar píxeles que ya han sido seleccionados
    mask = np.ones(len(sorted_coords), dtype=bool)
    
    # Aplicar supresión no máxima
    for i, coord in enumerate(sorted_coords):
        if mask[i]:  # Si no ha sido suprimido
            filtered_coords.append(coord)
            # Suprimir las coincidencias dentro del área de la plantilla
            supression_area = (
                (sorted_coords[:, 0] > coord[0] - suppression_radius_y) &
                (sorted_coords[:, 0] < coord[0] + suppression_radius_y) &
                (sorted_coords[:, 1] > coord[1] - suppression_radius_x) &
                (sorted_coords[:, 1] < coord[1] + suppression_radius_x)
            )
            mask[supression_area] = False  # Suprimir las coordenadas cercanas
    
    return np.array(filtered_coords)

def get_best_in_region(match_values, coords, template_shape, num_template):
    """
    Aplica supresión no máxima para eliminar coincidencias cercanas que representen la misma región.
    
    Parameters:
    - match_values: Array de coincidencias por rotación.
    - coords: Coordenadas de los píxeles seleccionados.
    - template_shape: Tamaño de la plantilla (alto, ancho).
    - num_template: Numero de la tempalte que estamos matcheando
    Returns:
    - filtered_coords: Coordenadas de las mejores coincidencias después de la supresión no máxima.
    """
    # Ordenar las coordenadas en función del valor máximo de coincidencia en cada coordenada (de mayor a menor)
    max_per_pixel = np.max(match_values, axis=2)  # Obtener los valores máximos para cada pixel
    sorted_indices = np.argsort(max_per_pixel[coords[:, 0], coords[:, 1]])[::-1]  # Ordenar de mayor a menor
    sorted_coords = coords[sorted_indices]  # Coordenadas ordenadas por la mejor coincidencia
    
    # Inicializar lista de coordenadas filtradas
    filtered_coords = []
    
    # Tamaño de la ventana de supresión (basada en el tamaño de la plantilla)
    suppression_radius_y, suppression_radius_x = template_shape[0] // 4, template_shape[1] // 4
    
    # Marcar píxeles que ya han sido seleccionados
    mask = np.ones(len(sorted_coords), dtype=bool)
    
    # Aplicar supresión no máxima
    for i, coord in enumerate(sorted_coords):
        if mask[i]:
            next_coord = coord
            aux = np.argmax(match_values[coord[0]][coord[1]])
            print("Angulo máximo sin correlacion: ", aux)
            # Aplicar la función a todas las coordenadas cercanas
            nearby_coords = sorted_coords[
                (sorted_coords[:, 0] > next_coord[0] - suppression_radius_y) &
                (sorted_coords[:, 0] < next_coord[0] + suppression_radius_y) &
                (sorted_coords[:, 1] > next_coord[1] - suppression_radius_x) &
                (sorted_coords[:, 1] < next_coord[1] + suppression_radius_x)
            ]
            
            # Aplicar la función concreta a cada coordenada cercana
            corrs = [get_correlation(match_values, coord, num_template) for coord in nearby_coords]
            print("Correlacion maxima: ", np.max(corrs))
            maxis = [get_maximum(match_values, coord) for coord in nearby_coords]
            maxi_corr = np.max(corrs)
            proporcion = 1/3
            total = [proporcion*v+ (1-proporcion)*m for v, m in zip(corrs, maxis)]
            # Obtener la coordenada con el mayor valor
            max_value_index = np.argmax(total)
            print(np.max(total))
            print("Maximo: ",maxis[max_value_index])
            print("Corr: ",corrs[max_value_index])

            next_coord = nearby_coords[max_value_index]
            filtered_coords.append(next_coord)
            
            # Suprimir las coincidencias dentro del área de la plantilla
            supression_area = (
                (sorted_coords[:, 0] > next_coord[0] - suppression_radius_y) &
                (sorted_coords[:, 0] < next_coord[0] + suppression_radius_y) &
                (sorted_coords[:, 1] > next_coord[1] - suppression_radius_x) &
                (sorted_coords[:, 1] < next_coord[1] + suppression_radius_x)
            )
            mask[supression_area] = False  # Suprimir las coordenadas cercanas
        
    
    return np.array(filtered_coords)

def get_maximum(match_values, coord):
    return max(match_values[coord[0]][coord[1]])

def get_correlation(match_values, coord, num_template):
    """
    Obtiene el valor de correlación para una coordenada específica y una plantilla específica.
    
    Parameters:
    - match_values: Array de coincidencias por rotación.
    - coord: Coordenada del píxel (x, y).
    - num_template: Número de la plantilla que estamos matcheando.
    
    Returns:
    - value: Valor de correlación para la coordenada y plantilla especificada.
    """
    file_path = './FuncionesReferencia/referencias.csv'
    with open (file_path, 'r') as f:
        rows = list(csv.reader(f, delimiter=';'))
        specific_row = rows[num_template]
        funcion_org = [float(value) for value in specific_row]
    
    x, y = coord[0], coord[1]
    #print(coord)
    funcion_comp = match_values[x][y]
    indice_max = np.argmax(funcion_comp)
    funcion_comp = np.roll(funcion_comp, -indice_max)  
    

    func1 = funcion_org
    func2 = funcion_comp

   
    # Calcular la correlación de Pearson
  
    corr_matrix = np.corrcoef(funcion_org, funcion_comp)
    corr_coef = corr_matrix[0, 1]
    if x== 322 and y ==53:
        print("Coeficiente en 56,50: ", corr_coef)
    return corr_coef

   
def suavizar_match_values(match_values, sigma=(1, 1, 3)):
    """
    Aplica un suavizado gaussiano al volumen de coincidencias (X, Y, 360)
    """
    # Suavizado del volumen 3D
    match_suavizado = gaussian_filter(match_values, sigma=sigma)
    return match_suavizado

def graficar_valores_por_rotacion1(match_values, coords, num_rotations=360):
    """
    Grafica los valores de coincidencia para cada ángulo (0-359 grados) para los píxeles seleccionados.
    
    Parameters:
    - match_values: Array de coincidencias por rotación.
    - coords: Coordenadas de los píxeles seleccionados.
    - num_rotations: Número de rotaciones a mostrar en la gráfica.
    """
    angles = np.arange(num_rotations)  # Ángulos de 0 a 359 grados
    fig, axes = plt.subplots(len(coords) // 2 + len(coords) % 2, 2, figsize=(15, 7 * (len(coords) // 2 + len(coords) % 2)))
    axes = axes.flatten()  # Aplanar el array de ejes para iterar fácilmente
    # Graficar los valores de coincidencia para cada píxel
    for ax, coord in zip(axes, coords):        
        pixel_y, pixel_x = coord  # Obtener la coordenada del píxel
        
        # Obtener los valores de coincidencia para todas las rotaciones de ese píxel
        pixel_values = match_values[pixel_y, pixel_x]
        
        # Crear el gráfico de líneas
        # ax.figure(1)
        ax.plot(angles, pixel_values, label=f'Pixel ({pixel_x}, {pixel_y})')
        ax.set_title(f'Coincidencia por rotación - Pixel ({pixel_x}, {pixel_y})')
        ax.set_xlabel('Ángulo de rotación (grados)')
        ax.set_ylabel('Valor de coincidencia')
        ax.legend()
        ax.grid(True)
        # ax.show(block=False)
    plt.tight_layout()
    plt.show(block=False)   


def estimar_instancias_con_cluster_angular(match_values, threshold=0.25, 
                                           distancia_maxima_espacial=5, min_samples_espacial=5,
                                           distancia_maxima_angular_grados=25, min_samples_angular=3):
    """
    Detecta instancias y agrupa los valores angulares cercanos dentro de cada instancia.
    Se queda con el grupo angular más poblado por instancia y calcula su ángulo medio.

    Parámetros:
    - match_values: array (X, Y, 360)
    - threshold: umbral de coincidencia mínima
    - distancia_maxima_espacial: distancia máxima en píxeles entre miembros de un clúster
    - min_samples_espacial: mínimo de puntos por clúster espacial
    - distancia_maxima_angular_grados: máxima distancia angular entre miembros de un clúster
    - min_samples_angular: mínimo de puntos por clúster angular

    Retorna:
    - lista de diccionarios con claves: x, y, angulo, tamaño, pixeles_cluster
    """
    X, Y, A = match_values.shape

    # Paso 1: puntos por encima del threshold
    max_vals = match_values.max(axis=2)
    mask = max_vals > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("No hay coincidencias por encima del umbral.")
        return []

    # Paso 2: ángulo de mayor coincidencia por punto
    best_angles = np.argmax(match_values[mask], axis=1)
    angle_rad = best_angles * np.pi / 180
    angle_vec = np.c_[np.cos(angle_rad), np.sin(angle_rad)]  # (N, 2)

    # Paso 3: clustering espacial con AgglomerativeClustering
    if len(coords) < min_samples_espacial:
        return []
    
    clustering_spatial = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distancia_maxima_espacial,
        linkage="single"
    )
    spatial_labels = clustering_spatial.fit_predict(coords)

    resultado = []
    for label in np.unique(spatial_labels):
        idx = spatial_labels == label
        if np.sum(idx) < min_samples_espacial:
            continue

        puntos = coords[idx]
        angulos = best_angles[idx]
        vectores = angle_vec[idx]

        # Paso 4: clustering angular con distancia máxima entre miembros
        if len(vectores) < min_samples_angular:
            continue

        clustering_angular = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=2 * np.sin(np.radians(distancia_maxima_angular_grados) / 2),
            linkage="complete"
        )
        angular_labels = clustering_angular.fit_predict(vectores)

                # Buscar clúster angular más numeroso con score ponderado
        cluster_ids = np.unique(angular_labels[angular_labels != -1])
        if len(cluster_ids) == 0:
            continue  # sin clúster válido

        best_score = -np.inf
        best_cluster = None
        for cid in cluster_ids:
            idx_cluster = angular_labels == cid
            puntos_cluster = puntos[idx_cluster]
            if len(puntos_cluster) < min_samples_angular:
                continue

            correlaciones = max_vals[puntos_cluster[:, 0], puntos_cluster[:, 1]]
            score = len(correlaciones) * np.mean(correlaciones)
            if score > best_score:
                best_score = score
                best_cluster = cid

        if best_cluster is None:
            continue

        idx_angular = angular_labels == best_cluster



        angulos_validos = angulos[idx_angular]
        if len(angulos_validos) == 0:
            continue

        rad = angulos_validos * np.pi / 180
        x = np.cos(rad).mean()
        y = np.sin(rad).mean()
        angulo_promedio = (np.arctan2(y, x) * 180 / np.pi) % 360
        centroide = np.round(puntos[idx_angular].mean(axis=0)).astype(int)

        resultado.append({
            "x": centroide[0],
            "y": centroide[1],
            "angulo": int(np.round(angulo_promedio)),
            "tamaño": len(angulos_validos),
            "pixeles_cluster": [tuple(p) for p in puntos[idx_angular]]
        })

    return resultado

def imprimir_clusters_detectados(resultados):
    """
    Muestra por pantalla los resultados de los clusters angulares detectados.

    Parameters:
    - resultados: lista de dicts con claves 'posicion', 'angulo', 'pixeles_cluster'
    """
    if not resultados:
        print("No se detectaron clusters.")
        return

    for i, r in enumerate(resultados):
        print(f"Cluster #{i+1}:")
        print(f"  - Ángulo medio: {r['angulo']:.2f}°")
        print(f"  - Posición media: ({r['x']:.2f}, {r['y']:.2f})")
        print(f"  - Número de píxeles: {len(r['pixeles_cluster'])}")
        print("  - Píxeles del cluster:")
        for p in r['pixeles_cluster']:
            print(f"    ({p[0]}, {p[1]})")
        print("-" * 40)

def encontrar_maximo_con_score_combinado(match_values,
                                          alpha=1.0, 
                                          beta=1.0, 
                                          gamma=1.0, 
                                          neighborhood_size=3):
    """
    Encuentra el mejor ángulo combinando el valor de coincidencia, 
    la coherencia local y la densidad angular.

    Parámetros:
    - match_values: np.ndarray de forma (X, Y, 360)
    - alpha, beta, gamma: pesos de cada componente del score
    - neighborhood_size: tamaño de la vecindad para calcular coherencias

    Retorna:
    - score
    """
    X, Y, A = match_values.shape

    # Componente 1: valor original de coincidencia (match_value)
    match_norm = match_values / match_values.max()

    # Componente 2: coherencia local (promedio local en vecindad)
    coherence_local = np.zeros_like(match_values)
    for angle in range(A):
        coherence_local[..., angle] = uniform_filter(
            match_norm[..., angle], size=neighborhood_size, mode='constant'
        )

    # Componente 3: coherencia angular (valores en ángulos cercanos)
    coherence_angular = np.zeros_like(match_values)
    for angle in range(A):
        vecinos = [(angle - 1) % A, angle, (angle + 1) % A]
        coherence_angular[..., angle] = np.mean(match_norm[..., vecinos], axis=-1)

    # Score final ponderado
    score = (
        alpha * match_norm +
        beta * coherence_local +
        gamma * coherence_angular
    )
    # Normalizar el score al rango 0 - match_values.max
    score = score / score.max() * match_values.max()

    
    return score

def get_max_coords(match_values, template, threshold):
    max_per_pixel = np.max(match_values, axis=2)  # Obtener el máximo valor de coincidencia para cada píxel
    max_coords = np.argwhere(max_per_pixel >= threshold)  # Coordenadas donde el máximo supera el threshold
    filtered_coords = supresion_no_maxima(match_values, max_coords, template.shape, radius=4)
    return filtered_coords

def graficar_valores_por_rotacion(match_values, coords, matriz_angulos, num_rotations=360):
    """
    Grafica los valores de coincidencia para cada ángulo (0-359 grados) para los píxeles seleccionados.
    
    Parameters:
    - match_values: Array de coincidencias por rotación.
    - coords: Coordenadas de los píxeles seleccionados.
    - matriz_angulos: Matriz con los ángulos originales de cada píxel.
    - num_rotations: Número de rotaciones a mostrar en la gráfica.
    """
    angles = np.arange(num_rotations)  # Ángulos de 0 a 359 grados
    
    # Crear una figura con subplots
    fig, axes = plt.subplots(len(coords) // 2 + len(coords) % 2, 2, figsize=(15, 7 * (len(coords) // 2 + len(coords) % 2)))
    axes = axes.flatten()  # Aplanar el array de ejes para iterar fácilmente
    
    # Graficar los valores de coincidencia para cada píxel
    for ax, coord in zip(axes, coords):
        pixel_y, pixel_x = coord  # Obtener la coordenada del píxel
        
        # Obtener los valores de coincidencia para todas las rotaciones de ese píxel
        pixel_values = match_values[pixel_y, pixel_x]
        
        original_angle = get_original_angle(matriz_angulos, coord)        
        # Crear el gráfico de líneas
        ax.scatter(np.argmax(pixel_values), np.max(pixel_values), color='blue', zorder=5)
        ax.scatter(original_angle, pixel_values[int(original_angle)], color='red', zorder=5)
        ax.plot(angles, pixel_values, label=f'Pixel ({pixel_x}, {pixel_y})')
        ax.set_title(f'Coincidencia por rotación - Pixel ({pixel_x}, {pixel_y})')
        ax.set_xlabel('Ángulo de rotación (grados)')
        ax.set_ylabel('Valor de coincidencia')
        ax.legend()
        ax.grid(True)
    
    # Eliminar subplots vacíos si hay un número impar de coordenadas
    if len(coords) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.show(block=False)      
        
def match_template_con_rotaciones(image, template, num_template=1,num_rotations=360, threshold=0.8, type="supresion"):
    """
    Aplica match_template con múltiples rotaciones de la plantilla y devuelve un array
    con 360 valores posibles de coincidencia por cada píxel.
    
    Parameters:
    - image: La imagen donde se buscan coincidencias.
    - template: La plantilla a usar para la coincidencia.
    - num_rotations: Número de rotaciones a aplicar (ej. 360 para 1 grado por rotación).
    - threshold: Umbral para filtrar coincidencias.
    
    Returns:
    - match_values: Un array donde para cada píxel hay una lista con las coincidencias
      correspondientes a cada rotación.
    - max_coords: Coordenadas de los píxeles donde al menos una rotación supera el threshold.
    """
    # Dimensiones de la imagen
    img_shape = image.shape
    template_shape = template.shape
    max_val = image.max()
    min_val = image.min()
    image= normalizar_matriz(image)
    
    # Crear un array para almacenar los valores de coincidencia para cada pixel
    match_values = np.zeros((img_shape[0], img_shape[1], num_rotations))
    
    # Aplicar las rotaciones y calcular match_template para cada ángulo
    for angle in range(num_rotations):
        # Rotar la plantilla
        template_rotada = reduce_matrix(template)
        
        template_rotada = rotate(template_rotada, angle, resize=False, mode='constant', cval=0)
        template_rotada = normalizar_matriz(template_rotada)
        

        # io.imsave(f'Template/template_rotada_{angle}.png', template_rotada)

        #quitamos todos los pixeles blancos que podemos        
        

        
        
        

        # Guardar la plantilla rotada en la carpeta Template con el nombre del ángulo correspondiente
        # if(angle == 45):
        #     matriz_a_imagen_gris1(template_rotada)
        # if (angle == 0):
        #     plt.imshow(template_rotada, cmap='gray')
        #     plt.title(f'Template rotado {angle}°')
        #     plt.show()
        # Aplicar match_template para la plantilla rotada
        match_result = match_template(image,template_rotada)
        
        # Almacenar el resultado de coincidencia en el array de match_values
        height, width = match_result.shape
        match_values[:height, :width, angle] = match_result
    
    # Filtrar los píxeles donde al menos una rotación tiene un valor superior al threshold
    max_per_pixel = np.max(match_values, axis=2)  # Obtener el máximo valor de coincidencia para cada píxel
    max_coords = np.argwhere(max_per_pixel >= threshold)  # Coordenadas donde el máximo supera el threshold
    
    # Aplicar supresión no máxima para evitar duplicados cercanos
    if type=="supresion":
        filtered_coords = supresion_no_maxima(match_values, max_coords, template.shape, radius=2)
    else:
        filtered_coords = get_best_in_region(match_values, max_coords, template.shape, num_template=num_template)
    #filtered_coords=max_coords
    return match_values, filtered_coords

def match_template_con_rotaciones_mask(image, template, num_template=1,num_rotations=360, threshold=0.8, type="supresion"):
    """
    Aplica match_template con múltiples rotaciones de la plantilla y devuelve un array
    con 360 valores posibles de coincidencia por cada píxel.
    
    Parameters:
    - image: La imagen donde se buscan coincidencias.
    - template: La plantilla a usar para la coincidencia.
    - num_rotations: Número de rotaciones a aplicar (ej. 360 para 1 grado por rotación).
    - threshold: Umbral para filtrar coincidencias.
    
    Returns:
    - match_values: Un array donde para cada píxel hay una lista con las coincidencias
      correspondientes a cada rotación.
    - max_coords: Coordenadas de los píxeles donde al menos una rotación supera el threshold.
    """
    # Dimensiones de la imagen
    img_shape = image.shape
    template_shape = template.shape
    max_val = image.max()
    min_val = image.min()
    image= normalizar_matriz(image)
    
    # Crear un array para almacenar los valores de coincidencia para cada pixel
    match_values = np.zeros((img_shape[0], img_shape[1], num_rotations))
    
    # Aplicar las rotaciones y calcular match_template para cada ángulo
    for angle in range(num_rotations):
        # Rotar la plantilla
        template_rotada = reduce_matrix(template)
        
        template_rotada = rotate(template_rotada, angle, resize=False, mode='constant', cval=0)
        template_rotada = normalizar_matriz(template_rotada)
        mask = (template_rotada <256).astype(np.uint8)
        
        # if angle==45:
        #     print(mask)
        #     print(template_rotada)

        # io.imsave(f'Template/template_rotada_{angle}.png', template_rotada)

        #quitamos todos los pixeles blancos que podemos        
        

        
        
        

        # Guardar la plantilla rotada en la carpeta Template con el nombre del ángulo correspondiente
        # if(angle == 45):
        #     matriz_a_imagen_gris1(template_rotada)
        # if (angle == 0):
        #     plt.imshow(template_rotada, cmap='gray')
        #     plt.title(f'Template rotado {angle}°')
        #     plt.show()
        # Aplicar match_template para la plantilla rotada
        metodo = cv2.TM_CCORR_NORMED
        match_result = cv2.matchTemplate(image, template_rotada, mask = mask, method=metodo)
        
        # Almacenar el resultado de coincidencia en el array de match_values
        height, width = match_result.shape
        match_values[:height, :width, angle] = match_result
    
    # Filtrar los píxeles donde al menos una rotación tiene un valor superior al threshold
    max_per_pixel = np.max(match_values, axis=2)  # Obtener el máximo valor de coincidencia para cada píxel
    max_coords = np.argwhere(max_per_pixel >= threshold)  # Coordenadas donde el máximo supera el threshold
    
    # Aplicar supresión no máxima para evitar duplicados cercanos
    if type=="supresion":
        filtered_coords = supresion_no_maxima(match_values, max_coords, template.shape, radius=4)
    else:
        filtered_coords = get_best_in_region(match_values, max_coords, template.shape, num_template=num_template)
    #filtered_coords=max_coords
    return match_values, filtered_coords

        
        
def pad_template(template, padding_size):
    """
    Agrega un padding de ceros alrededor de la plantilla para evitar que se corte tras la rotación.
    """
    pad_height = 0
    pad_width = padding_size
    return np.pad(template, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=1)

def obtener_mejor_angulo_por_pixel(match_values, coords):
    """Obtiene el ángulo de rotación con la mejor coincidencia para cada píxel."""
    best_angles = []
    for coord in coords:
        y, x = coord
        # Obtener el ángulo con el valor máximo para cada coordenada
        best_angle = np.argmax(match_values[y, x])
        best_angles.append(best_angle)
    return np.array(best_angles)

def obtener_mejor_angulo_y_valor_por_pixel(match_values, coords):
    """Obtiene el ángulo de rotación con la mejor coincidencia y su valor para cada píxel."""
    best_angles = []
    best_values = []
    for coord in coords:
        y, x = coord
        # Obtener el ángulo con el valor máximo para cada coordenada
        best_angle = np.argmax(match_values[y, x])
        best_value = match_values[y, x, best_angle]
        best_angles.append(best_angle)
        best_values.append(best_value)
    return np.array(best_angles), np.array(best_values)



def get_original_angle(matriz_angulos, coord):
    # Obtener las coordenadas de los elementos distintos de cero
    coordinates = np.argwhere(matriz_angulos != 0)

    # Convertir a una lista de tuplas (x, y)
    coordinates_tuples = [tuple(coord) for coord in coordinates]
    closest_coord, min_distance = get_closest_coord(coord, coordinates_tuples)
    x= closest_coord[0]
    y = closest_coord[1]
    original_angle = matriz_angulos[x,y]
    return original_angle
 
def check_results(matriz_angulos, max_coords, best_angles, num_rotations):
    # Obtener las coordenadas de los elementos distintos de cero
    coordinates = np.argwhere(matriz_angulos != 0)

    # Convertir a una lista de tuplas (x, y)
    coordinates_tuples = [tuple(coord) for coord in coordinates]
    
    i=0
    if len(max_coords) == 0:
        print("No se encontraron coincidencias.")
    else:
        for coord in max_coords:
        
            closest_coord, min_distance = get_closest_coord(coord, coordinates_tuples)
            x= closest_coord[0]
            y = closest_coord[1]
            original_angle = matriz_angulos[x,y]
            matched_angle = best_angles[i]
            print("-----Coordenadas "+str(coord)+"-------")
            print("Coordenadas originales: (" + str(int(closest_coord[0])) + ", " + str(int(closest_coord[1])) + ")")
            print("Angulo original: "+str(original_angle))
            print("Angulo matcheado: "+ str(matched_angle))
            print("--")
            i=i+1

def get_closest_coord(target_coord, coordinates_values):
    """
    Encuentra la coordenada más cercana en coordinates_values a target_coord.

    Args:
        target_coord (tuple): Coordenada objetivo (x, y).
        coordinates_values (list): Lista de tuplas (x, y).

    Returns:
        tuple: Coordenada más cercana (x, y).
    """
    closest_coord = None
    min_distance = float('inf')  # Inicializar con un valor grande
    for coord in coordinates_values:
        x, y = coord  # Ignorar el valor y solo trabajar con las coordenadas
        distance = np.sqrt((x - target_coord[0]) ** 2 + (y - target_coord[1]) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_coord = (x, y)

    return closest_coord, min_distance


def dibujar_rectangulos_rotados(image, template, coords, best_angles):
    """
    Dibuja rectángulos rotados sobre la imagen original donde se encuentren las coincidencias.
    
    Parameters:
    - image: La imagen original.
    - template: La plantilla (para determinar el tamaño de los rectángulos).
    - coords: Coordenadas de las coincidencias detectadas.
    - best_angles: Los mejores ángulos de rotación para cada coincidencia.
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')  # Mostrar la imagen original en escala de grises

    

    for coord, angle in zip(coords, best_angles):
        y, x = coord
        # Crear el rectángulo (sin rotación aún)
        template_rotada = rotate(template, angle, resize=False, mode='constant', cval=0)
        #quitamos todos los pixeles blancos que podemos
        template_rotada = reduce_matrix(template_rotada)
        template_height, template_width = template_rotada.shape
        rect = Rectangle((x, y), template_width, template_height, edgecolor='red', facecolor='none', lw=2)

        # Rotar el rectángulo usando la mejor rotación encontrada
        t = Affine2D().rotate_deg_around(x + template_width / 2, y + template_height / 2, angle)
        rect.set_transform(t + ax.transData)

        ax.add_patch(rect)  # Añadir el rectángulo a la imagen

    plt.show()
    
def dibujar_rectangulos_rotados_color(image, template, coords, best_angles, best_values):
    """
    Dibuja rectángulos rotados sobre la imagen original donde se encuentren las coincidencias.
    
    Parameters:
    - image: La imagen original.
    - template: La plantilla (para determinar el tamaño de los rectángulos).
    - coords: Coordenadas de las coincidencias detectadas.
    - best_angles: Los mejores ángulos de rotación para cada coincidencia.
    - best_values: Los mejores valores de coincidencia para cada coordenada.
    """
    
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')  # Mostrar la imagen original en escala de grises

    # Normalizar los valores de coincidencia para que estén en el rango [0, 1]
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])

    for coord, angle, value in zip(coords, best_angles, best_values):
        y, x = coord
        # Crear el rectángulo (sin rotación aún)
        template_rotada = rotate(template, angle, resize=False, mode='constant', cval=0)
        template_rotada = reduce_matrix(template_rotada)
        template_height, template_width = template_rotada.shape
        
        # Escalar el valor a un rango de colores (0 a 1)
        color_value = sm.to_rgba(value)
        
        rect = Rectangle((x, y), template_width, template_height, edgecolor=color_value, facecolor='none', lw=2)

        # Rotar el rectángulo usando la mejor rotación encontrada
        t = Affine2D().rotate_deg_around(x + template_width / 2, y + template_height / 2, angle)
        rect.set_transform(t + ax.transData)

        ax.add_patch(rect)  # Añadir el rectángulo a la imagen

        # Añadir el valor de coincidencia al lado del rectángulo
        ax.text(x, y - 10, f'{value:.2f}', color='white', fontsize=8, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5))

    # Añadir la barra de color
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Valor de coincidencia')

    plt.show(block=False)
    



def dibujar_max_rotacion(match_values, threshold=0.28):
    """
    Dibuja una imagen donde cada píxel representa el ángulo con el valor máximo de coincidencia,
    con una escala de color de rojo (0°) a verde (360°). Los píxeles cuyo valor máximo no supere
    el umbral se dejan en blanco. Se muestra también una leyenda de color.
    
    Parameters:
    - match_values: Array de forma (X, Y, 360) con los valores de coincidencia por rotación.
    - threshold: Valor mínimo de coincidencia para colorear el píxel.
    """
    # Obtener ángulo de máxima coincidencia y su valor
    angulos_maximos = np.argmax(match_values, axis=2)
    valores_maximos = np.max(match_values, axis=2)

    # Normalizar ángulos [0, 359] → [0, 1] para interpolación de color
    t = angulos_maximos / 359.0
    r = 1.0 - t
    g = t
    b = np.zeros_like(t)

    # Imagen RGB
    rgb_image = np.stack([r, g, b], axis=2)

    # Aplicar máscara para dejar en blanco los valores bajos
    mascara = valores_maximos < threshold
    rgb_image[mascara] = [1.0, 1.0, 1.0]

    # Mostrar imagen principal
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rgb_image)
    ax.set_title(f"Ángulo de máxima coincidencia (Rojo=0° → Verde=360°)\nThreshold = {threshold}")
    ax.axis('off')

    # Crear barra de color personalizada
    n = 360
    cmap = ListedColormap([[1 - i / (n - 1), i / (n - 1), 0] for i in range(n)])
    norm = Normalize(vmin=0, vmax=359)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # posición de la barra
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cbar.set_label('Ángulo (grados)')

    plt.show()
    
def dibujar_max_rotacion_numeros(match_values, threshold=0.25):
    """
    Muestra una cuadrícula con los valores de ángulo (en grados) en cada celda,
    si el valor de coincidencia máxima supera el umbral dado.
    
    Parameters:
    - match_values: Array de forma (X, Y, 360) con los valores de coincidencia por rotación.
    - threshold: Umbral mínimo para mostrar el ángulo en una celda.
    """
    # Obtener ángulos y valores máximos
    angulos_maximos = np.argmax(match_values, axis=2)
    valores_maximos = np.max(match_values, axis=2)

    h, w = angulos_maximos.shape

    # Crear imagen en blanco para la cuadrícula
    grid_image = np.ones((h, w, 3))  # blanco de fondo

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(grid_image)
    ax.set_title(f"Ángulos de máxima coincidencia (solo si > {threshold})")
    ax.set_xticks(np.arange(w))
    ax.set_yticks(np.arange(h))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='gray', linewidth=0.5)

     # Mostrar texto en cada celda
    for y in range(h):
        for x in range(w):
            if valores_maximos[y, x] >= threshold:
                angulo = angulos_maximos[y, x]
                ax.text(x, y - 0.2, f"{angulo}°", ha='center', va='center', fontsize=6, color='black')
                ax.text(x, y + 0.3, f"({x},{y})", ha='center', va='center', fontsize=5.5, color='gray')

    ax.tick_params(length=0)  # sin marcas de eje
    plt.show()

    


    
