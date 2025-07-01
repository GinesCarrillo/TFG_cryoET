import torch
import numpy as np
import os
import pandas as pd
from build_weighted_complex import build_weighted_complex, plot_complex
from complex_to_weighted_ECT import complex_to_weighted_ECT
from utils_mod.createImage import load_mrc, create_image_train
from utils_mod.rotation_template_matching import normalizar_matriz

# ---------------- CONFIG ----------------
csv_path = "/Users/salvaromero/Desktop/Gines/swect6/labels.csv"
root_dir = "/Users/salvaromero/Desktop/Gines/swect6/swect_dataset"
img_dir = "/Users/salvaromero/Desktop/Gines/swect6/img_dataset"
SAVE_DIR = root_dir
DIR_TEMPLATES = "../templates"
TEMPLATE_TO_MATCH = 1
WIDTH = 27
HEIGHT = 27
NUM_NOISE_INSTANCES = 100
NUM_DIRECTIONS = 360
NUM_STEPS = 64
EPSILON = 4
os.makedirs(SAVE_DIR, exist_ok=True)

# -------- Leer CSV existente si lo hay --------
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if not df.empty:
        last_id = df['filename'].str.extract(r'sample_(\d+)')[0].astype(int).max()
        sample_id = last_id + 1
    else:
        sample_id = 0
else:
    sample_id = 0
    with open(csv_path, "w") as f:
        f.write("filename,label\n")

# -------- Cargar plantillas MRC --------
templatesName = ["3j9i_tilt_0_vsize_10_SNR_Inf.mrc", "4v4r_tilt_0_vsize_10_SNR_Inf.mrc"]
listTemplates = [load_mrc(os.path.join(DIR_TEMPLATES, t)) for t in templatesName]





# -------- Generación de datos --------
for angle in range(360):
    noise_levels = np.arange(0, 25.5, 0.25)
    for noise_level in noise_levels:        
        image, angle = create_image_train((WIDTH, HEIGHT), listTemplates, TEMPLATE_TO_MATCH, angle, noise_level,EPSILON)
        image = normalizar_matriz(image)
        image = 1-image/255.0  # Invertir y normalizar la imagen

        # Construcción del complejo simplicial
        V, E, F, Vw, Ew, Fw = build_weighted_complex(image)
        complex = {
            'V': V, 'E': E, 'F': F,
            'V_weights': Vw,
            'E_weights': Ew,
            'F_weights': Fw
        }

        # Cálculo del SWECT
        swect = complex_to_weighted_ECT(
            complex,
            num_directions=NUM_DIRECTIONS,
            num_steps=NUM_STEPS,
            method='gaussian',
            window=3,
            normalization_method='total'
        )

        # Guardar resultados
        fname = f"sample_{sample_id:05d}.pt"
        torch.save(swect, os.path.join(SAVE_DIR, fname))

        with open(csv_path, "a") as f:
            f.write(f"{fname},{angle}\n")

        # print(f"✅ Guardado {fname} con ángulo {angle}°")
        sample_id += 1

