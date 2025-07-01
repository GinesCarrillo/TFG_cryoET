import os
import numpy as np
import pandas as pd
from skimage.io import imread
from tqdm import tqdm
from predictor import SWECTPredictor
from WECT.build_weighted_complex import *  
from WECT.complex_to_weighted_ECT import *

SAVE_DIR = "/mnt/usb/FINAL/imgsPrueba"
LABELS_CSV = "/mnt/usb/FINAL/labelsPrueba.csv"
LABELS_CSV_OUT = "/mnt/usb/FINAL/labels_with_predictionsPrueba.csv"

SUBIMG_SIZE = 27
NUM_DIRECTIONS = 360
NUM_STEPS = 64

def extract_subimage(full_img, x, y, size):
    x = int(x)
    y = int(y)
    h, w = full_img.shape
    if y + size > h or x + size > w:
        raise ValueError(f"Subimagen fuera de límites: {x},{y} en imagen {w}x{h}")
    return full_img[y:y+size, x:x+size]

def main():
    df = pd.read_csv(LABELS_CSV).copy()  # Copia del DataFrame
    df["predicted_angle"] = np.nan        # Nueva columna

    predictor = SWECTPredictor("modelo_trigger-7.0-36k-32.pt")  # Cargar solo una vez

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filename = row["filename"].replace(".npy", ".png")
        x = row["x_pred"]
        y = row["y_pred"]

        img_path = os.path.join(SAVE_DIR, filename)
        image = imread(img_path, as_gray=True)
        try:
            subimg = extract_subimage(image, x, y, SUBIMG_SIZE)
        except ValueError as e:
            print(f"❌ Error en muestra {filename}: {e}")
            continue
        # print(subimg)

        subimg = 1 - subimg / 255.0 
        # import matplotlib.pyplot as plt
        # plt.imshow(subimg, cmap='gray')
        # plt.title(f"Subimagen: {filename} ({x},{y})")
        # plt.axis('off')
        # plt.show()
        V, E, F, Vw, Ew, Fw = build_weighted_complex(subimg)
        complex = {
            'V': V, 'E': E, 'F': F,
            'V_weights': Vw,
            'E_weights': Ew,
            'F_weights': Fw
        }
        # plot_complex(complex)

        swect = complex_to_weighted_ECT(
            complex,
            num_directions=NUM_DIRECTIONS,
            num_steps=NUM_STEPS,
            method='gaussian',
            window=3,
            normalization_method='total'
        )

        predicted_angle = predictor.predict(swect)
        df.at[idx, "predicted_angle"] = predicted_angle
        print(f"✅ {filename}: {predicted_angle:.2f}°")

    df.to_csv(LABELS_CSV_OUT, index=False)
    print(f"\n📁 Archivo con predicciones guardado en: {LABELS_CSV_OUT}")

if __name__ == "__main__":
    main()
