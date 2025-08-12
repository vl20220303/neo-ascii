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
    """
    Takes in a mask, converts it from 0-1 range to 0-255, and duplicates it across 3 channels to create an RGB image.
    Returns the image.
    """
    image = (mask * 255).astype(np.uint8) # scale to 0–255
    image = np.stack([image]*3, axis=-1)
    return image

def ascii_scaled(image, params=None):
    """
    Takes in an image and scales it to account for character aspect ratio.
    Returns the image.
    """
    height, width = image.shape[:2]

    scale_factor = 14/12
    if not params is None:
        font, font_scale, thickness, char_spacing, line_spacing = params
        scale_factor = line_spacing/char_spacing
    
    image = scale_image(image=image, new_size=(int(height/scale_factor), width), fit_type='cover')
    return image

def ascii_scaled_dims(height, width, params=None):
    """
    Takes in dimensions and scales it to account for character aspect ratio.
    Returns the dimensions.
    """
    scale_factor = 14/12
    if not params is None:
        font, font_scale, thickness, char_spacing, line_spacing = params
        scale_factor = line_spacing/char_spacing
    return (int(height/scale_factor), width)

def oversaturate(image, amt=None):
    """
    Takes in an image and adds {amt} to saturation.
        If amt is None, saturation of all pixels is set to max.
    Returns the image.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if amt is None:
        hsv_image[:, :, 1] = 255
    else:
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1].astype(int) + amt, 0, 255).astype(np.uint8) # avoids overflow
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image

def brighten(image, amt=None):
    """
    Takes in an image and adds {amt} to brightness.
        If amt is None, brightness of all pixels is set to max.
    Returns the image.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if amt is None:
        hsv_image[:, :, 2] = 255
    else:
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2].astype(int) + amt, 0, 255).astype(np.uint8) # avoids overflow
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return image

def greyscale(image):
    """
    Takes in an image and returns the greyscaled version.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255

def extension_type(filepath):
    """
    Returns the extension type of the file provided.
    """
    return os.path.splitext(filepath)[1].lower()