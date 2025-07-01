from utils_mod.lio import *
from PIL import Image
from utils_mod.reader import *
import napari


def normalizar_matriz(matriz):
    max_val = matriz.max()
    min_val = matriz.min()
    # Normalización
    matriz_normalizada = ((matriz - min_val) / (max_val - min_val)) * 255
    matriz_normalizada = matriz_normalizada.astype(np.uint8)  # Convertir a tipo entero sin signo de 8 bits
    return matriz_normalizada

def normalizar_template(matriz, maxi, mini):
    max_val = max(matriz.max(), maxi)
    min_val = min(matriz.min(), mini)
    # Normalización
    matriz_normalizada = ((matriz - min_val) / (max_val - min_val)) * 255
    matriz_normalizada = matriz_normalizada.astype(np.uint8)  # Convertir a tipo entero sin signo de 8 bits
    return matriz_normalizada   

def matriz_a_imagen_gris(matriz):
    
    # Convertir los valores de 0-1 a 0-255 para representar escala de grises
    matriz_escalada = normalizar_matriz(matriz)
    
    # Crear una imagen PIL a partir de la matriz
    imagen = Image.fromarray(matriz_escalada, mode='L')
    # imagen.show()
    return imagen

def matriz_a_imagen_gris1(matriz):    
    # Crear una imagen PIL a partir de la matriz
    imagen = Image.fromarray(matriz, mode='L')
    imagen.show()
    return imagen

def viewInNapari(path):
    # Start a Napari viewer
    viewer = napari.Viewer()

    # Open the MRC file (replace 'path_to_your_file.mrc' with your file path)
    viewer.open(path, plugin='napari-mrcfile-reader')

    # Start the Napari event loop if not in an interactive environment
    napari.run()


def showTemplate (temp):
    #Show an image of the template in the position num of the templates' list
    if len(temp.shape) >= 3 and temp.shape[2] == 1:
        np.set_printoptions(threshold=np.inf)
        matriz = temp[:, :, 0]
        #print(matriz)
        matriz_a_imagen_gris(matriz)
    else:
        print("It's a 3D image, I need a 2D image!")
    
def studyTemplate(temp):
    if len(temp.shape) >= 3 and temp.shape[2] == 1:
        matrix = temp[:, :, 0]
        maxi = matrix.max()
        mini = matrix.min()
        print(maxi,"-",mini)
        
    else:
        print("It's a 3D image, I need a 2D image!")
    
DIR_TEMPLATES = "./templates"
templatesName = ["3j9i_tilt_0_vsize_10_SNR_Inf.mrc","4v4r_tilt_0_vsize_10_SNR_Inf.mrc" ]

listTemplates = list()
for template in templatesName:
    listTemplates.append(load_mrc(DIR_TEMPLATES+"/"+template))

#viewInNapari(DIR_TEMPLATES+"/"+templatesName[0])
#showTemplate(listTemplates[0])
# for template in listTemplates:
#     showTemplate(template)