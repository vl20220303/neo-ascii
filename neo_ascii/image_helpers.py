import os
import numpy as np
import cv2

from neo_ascii.image_scaler import scale_image

def get_dims(image_dims, image, image_path):
    dims = ()
    if image_dims:
        dims = image_dims
    elif image is not None:
        height, width = image.shape[:2]
        dims = (height, width)
    elif image_path:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        dims = (height, width)
    else:
        raise ValueError("No dimension or image path provided!")
    return dims

def get_image(image, image_path, format):
    if image_path:
        image = cv2.imread(image_path)
    elif image is not None:
        pass
    else:
        raise ValueError("No image or image path provided!")

    if format == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif format == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Format must be either 'rgb' or 'hsv'!")
    return image

def mask_to_image(mask):
    image = (mask * 255).astype(np.uint8) # scale to 0–255
    image = np.stack([image]*3, axis=-1)
    return image

def ascii_scaled(image):
    height, width = image.shape[:2]
    image = scale_image(image=image, new_size=(int(height/1.1), width), fit_type='cover')
    return image

def ascii_scaled_dims(height, width):
    return (int(height/1.1), width)

def oversaturate(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv_image[:, :, 1] = 255  # Maximize Saturation
    hsv_image[:, :, 2] = 255  # Maximize Brightness (Value)

    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image

def extension_type(filepath):
    return os.path.splitext(filepath)[1].lower()