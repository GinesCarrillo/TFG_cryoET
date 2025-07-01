import numpy as np
from utils_mod.createImage import *
from utils_mod.rotation_template_matching import *
import matplotlib.pyplot as plt
import os
import csv

DIR_TEMPLATES = "./templates"
TEMPLATE_TO_MATCH = 1
THRESHOLD = 0.1
THRESHOLD_DIFF = 0.002
EPSILON = 4
WIDTH, HEIGHT = 100, 100
NUM_ROTATIONS = 360
TYPE = "supression"
NUM_SAMPLES_PER_LEVEL = 50

templatesName = ["3j9i_tilt_0_vsize_10_SNR_Inf.mrc", "4v4r_tilt_0_vsize_10_SNR_Inf.mrc"]
listTemplates = [load_mrc(os.path.join(DIR_TEMPLATES, t)) for t in templatesName]
template = listTemplates[TEMPLATE_TO_MATCH][:, :, 0]

errors_tm = []
errors_cluster = []

for noise_level in range(1, 76):
    total_error_tm = 0
    total_error_cluster = 0
    count = 0

    for _ in range(NUM_SAMPLES_PER_LEVEL):
        image, angle = create_image_train((WIDTH, HEIGHT), [0, 1], listTemplates, TEMPLATE_TO_MATCH, noise_level, eps=EPSILON)
        match_values, _ = match_template_con_rotaciones(image, template, num_rotations=NUM_ROTATIONS, threshold=THRESHOLD, type=TYPE)
        # Encontrar el índice del valor máximo en match_values
        max_idx = np.unravel_index(np.argmax(match_values), match_values.shape)
        y_max, x_max, angle_max = max_idx
        max_coords = (y_max, x_max, angle_max)

        best_angles, _ = obtener_mejor_angulo_y_valor_por_pixel(match_values, None)
        angulo_real = angles_matrix[angles_matrix > 0][0]

        # TM clásico: ángulo con mejor correlación
        angulo_tm = np.argmax(match_values.max(axis=(0, 1)))
        error_tm = min(abs(angulo_tm - angulo_real), 360 - abs(angulo_tm - angulo_real))
        total_error_tm += error_tm

        # Clustering angular
        clusters = estimar_instancias_con_cluster_angular2(match_values, threshold=THRESHOLD - THRESHOLD_DIFF)
        if clusters:
            angulo_cluster = clusters[0]['angulo']
            error_cluster = min(abs(angulo_cluster - angulo_real), 360 - abs(angulo_cluster - angulo_real))
            total_error_cluster += error_cluster
        else:
            total_error_cluster += 180  # Penalización por fallo
        
        print(f"  Muestra {_+1}: Real={angulo_real}, TM={angulo_tm} (error={error_tm}), Cluster={angulo_cluster if clusters else 'N/A'} (error={error_cluster if clusters else 'N/A'})")

        count += 1

    errors_tm.append(total_error_tm / count)
    errors_cluster.append(total_error_cluster / count)
    print(f"Nivel de ruido {noise_level}: TM={errors_tm[-1]:.2f}, Cluster={errors_cluster[-1]:.2f}")

csv_filename = "resultados_comparacion_tm_clustering.csv"
with open(csv_filename, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Nivel de ruido', 'Error TM', 'Error Clustering'])
    for noise_level, (err_tm, err_cluster) in enumerate(zip(errors_tm, errors_cluster), start=1):
        writer.writerow([noise_level, err_tm, err_cluster])
print(f"Resultados guardados en {csv_filename}")
# Graficar resultados
plt.plot(range(1, 76), errors_tm, label='TM tradicional')
plt.plot(range(1, 76), errors_cluster, label='Clustering angular')
plt.xlabel('Nivel de ruido')
plt.ylabel('Error angular medio (°)')
plt.title('Comparación de TM vs. clustering angular')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
