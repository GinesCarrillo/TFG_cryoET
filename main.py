import argparse
# from utils_mod.utils import *
from utils_mod.createImage import * 
from skimage.transform import rotate
from utils_mod.matrices import *
import random
from utils_mod.reduceMatrix import reduce_matrix, is_surrounded
from utils_mod.rotation_template_matching import *
# from tda import *   
import numpy as np

#--------------------------PARAMETERS------------------------------------------------

#Directory where the templates are stored
DIR_TEMPLATES = "./templates"

#Noise level to add to the image
NOISE_LEVEL=0 #Bigger value, less noise (Noise where rotations start to fail: 0.3 (THRESHOLD=0.3))

# THRESHOLD for the template matching
THRESHOLD = 0.9
THRESHOLD_DIFF = 0.002

#THRESHOLD to consider a value as zero
EPSILON = 4

#Number of instances of each template in the image
NUM_INSTANCES=[0,3]

#Select the template we want to match (0 caramelo, 1 la otra)
TEMPLATE_TO_MATCH = 1

#Shape of the image to create
WIDTH = 400
HEIGHT = 300

NUM_ROTATIONS = 360

TYPE = "correlation"   #supresion or correlation
#-----------------------------------------------------------------------------------


def main(image_path=None):
    templatesName = ["3j9i_tilt_0_vsize_10_SNR_Inf.mrc","4v4r_tilt_0_vsize_10_SNR_Inf.mrc" ]

    listTemplates = list()
    for template in templatesName:
        listTemplates.append(load_mrc(DIR_TEMPLATES+"/"+template))
        
    num_templates = len(listTemplates)

    arr_count =np.array(NUM_INSTANCES)

    template = listTemplates[TEMPLATE_TO_MATCH]

    #We only need the first two dimensions
    template = template[:, :, 0]

    if image_path:
        image = load_image(image_path)
        angles_matrix = None  # Assuming we don't have angles matrix for external images
    else:
        image, angles_matrix = create_image((WIDTH, HEIGHT), arr_count, listTemplates, TEMPLATE_TO_MATCH, NOISE_LEVEL, eps=EPSILON)
        matriz_a_imagen_gris(image)

    #We apply the template matching to the image
    # Call the function with 360 rotations
    match_values, max_coords = match_template_con_rotaciones(image, template, num_rotations=NUM_ROTATIONS, threshold=THRESHOLD, type=TYPE)
    # Create an image with the best correlation values for each pixel
    best_correlation_image = np.max(match_values, axis=2)

    # Display the image
    plt.imshow(best_correlation_image, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()
    plt.title("Best Correlation Values")
    max_coords = get_max_coords(match_values = match_values, template=template, threshold= THRESHOLD)
    # instancias = estimar_instancias_con_cluster_angular2(match_values=match_values, threshold=THRESHOLD-THRESHOLD_DIFF)
    # imprimir_clusters_detectados(instancias)
    print(max_coords.shape)
    if angles_matrix is not None:
        graficar_valores_por_rotacion(match_values, max_coords, matriz_angulos= angles_matrix,num_rotations=NUM_ROTATIONS)
        
        print("llego aqui")


    else:
        graficar_valores_por_rotacion1(match_values, max_coords,num_rotations=NUM_ROTATIONS)

    #Obtain the best angle and value for each pixel
    best_angles, best_values = obtener_mejor_angulo_y_valor_por_pixel(match_values, max_coords)

    #Check the results
    if angles_matrix is not None:
        check_results(angles_matrix, max_coords, best_angles, NUM_ROTATIONS)
    else :
        print(best_values)
        print(best_angles)

    # Draw rectangles with the best angles on the image
    dibujar_rectangulos_rotados_color(image, template, max_coords, best_angles, best_values)
    dibujar_max_rotacion_numeros(match_values, threshold=THRESHOLD-THRESHOLD_DIFF)
    #Diagrama de persistencia
    # theta = np.arange(0, 360)
    #compute_persistence_diagram2(theta, match_values, max_coords)


def load_image(image_path):
    # Implement this function to load the image from the given path
    # For example, using skimage.io.imread
    from skimage.io import imread
    return imread(image_path, as_gray=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image or create a new one.")
    parser.add_argument('--image', type=str, help='Path to the image file')
    args = parser.parse_args()

    main(args.image)