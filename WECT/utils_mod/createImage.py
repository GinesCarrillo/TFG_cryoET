
from skimage.transform import rotate
from utils_mod.matrices import *
import random
from utils_mod.reduceMatrix import reduce_matrix, is_surrounded
from utils_mod.rotation_template_matching import *

def addNoiseToTemplate(image, noise_level):
    mean = image.mean()
    sg_fg = abs(mean / noise_level)
    image = image + np.random.normal(mean, sg_fg, image.shape)
    return image

def addNoiseToImage(image, noise_level, mean):
    sg_fg = noise_level
    image = image + np.random.normal(0, sg_fg, image.shape)
    return image

def instantiate_template(template, image, angle, fixed=False, eps=1):
    # angle = int(random.uniform(0, 360))
    template = rotate(template, angle, resize=True,  mode='constant', cval=0)
    template = reduce_matrix(template)
    #mostrar_imagen(template)
    #template = addNoiseToTemplate(template, noise_level)
    height_img, width_img  = image.shape
    height_temp, width_temp  = template.shape
    if fixed:
        x = (height_img - height_temp) // 2
        y = (width_img - width_temp) // 2
    else:
        x = int(random.uniform(0, height_img - height_temp))
        y = int(random.uniform(0, width_img - width_temp))

    for i in range(height_temp):
        for j in range(width_temp):
            if (template[i,j].any() or is_surrounded(template,i,j)) and( abs(template[i,j]) >eps):
                image[x+i,y+j] = template[i,j]
    #image[x:x + width_temp, y:y + height_temp] = template
    return image, x, y, angle



def create_image(dimensions, arr_count, listTemplates, numTemplate, noise_level, eps=1):
    #dimensions: width and height of the image
    #arr_count: number of instances of each template in the image
    #listTemplates: list with all the templates we want to instantiate
    #numTemplate: template pos we're matching

    width, height = dimensions
    image = np.zeros((height, width))
    matriz_angulos = np.zeros((height, width))
    #Instantiating the templates in the image
    for i in range(len(arr_count)):
        for j in range(arr_count[i]):
            #Pasamos de matriz 3d a 2d
            aux = listTemplates[i]
            template = aux[:, :, 0]
            image,x,y, angle = instantiate_template(template, image, fixed=False, eps=eps)
            if i == numTemplate:
                matriz_angulos[x,y] = angle

    #Vamos a añadir ruido a la imagen a partir de la media de la plantilla que buscamos
    aux = listTemplates[numTemplate]
    template = aux[:, :, 0]
    template = reduce_matrix(template, eps=eps)
    mean = template.mean()
    image_with_noise = addNoiseToImage(image, noise_level, mean)
    return image_with_noise, matriz_angulos

def create_image_train(dimensions, listTemplates, numTemplate,angle, noise_level, eps=1):
    #dimensions: width and height of the image
    #arr_count: number of instances of each template in the image
    #listTemplates: list with all the templates we want to instantiate
    #numTemplate: template pos we're matching

    width, height = dimensions
    image = np.zeros((height, width))
    #Instantiating the templates in the image
    
    #Pasamos de matriz 3d a 2d
    aux = listTemplates[numTemplate]
    template = aux[:, :, 0]
    image,x,y, angle = instantiate_template(template, image,angle, fixed=True, eps=eps)

    #Vamos a añadir ruido a la imagen a partir de la media de la plantilla que buscamos
    aux = listTemplates[numTemplate]
    template = aux[:, :, 0]
    template = reduce_matrix(template, eps=eps)
    mean = template.mean()
    image_with_noise = addNoiseToImage(image, noise_level, mean)
    return image_with_noise, angle




