import cv2
import numpy as np

from neo_ascii.utils import get_image

def filter(image=None, image_path=None, format='hsv', include=[(0, 180), (0, 255), (0, 255)], activation='const', output_path=None):
    """
    Filters pixels of a given image or image path.
        Pixels must conform to one of 3 criteria to remain.
    format is the format of the criteria.
    include is the selection criteria, and includes 3 tuples (1 for each channel), each containing an upper and lower bound that wraps around.
    activation is the activation method of the mask.
        Includes 'const', 'linear', '2-sided-linear'
    """
    image = get_image(image, image_path, format)

    if activation not in {'const', 'linear', '2-sided-linear'}:
        raise ValueError("Activation must be 'const', 'linear', or '2-sided-linear'")

    mask = np.ones(image.shape[:2])

    for i, (low, high) in enumerate(include):
        channel = image[:, :, i]
        if low <= high:
            valid = (channel >= low) & (channel <= high)
        else:
            valid = (channel >= low) | (channel <= high)

        mask *= valid

        if activation == 'linear':
            norm = (channel - low) / (high - low)
            norm = np.clip(norm*2, 0.5, 1)
            mask *= norm

        elif activation == '2-sided-linear':
            norm = (channel - low - (high-low)/2) / (high - low)
            mask *= norm

    if activation == 'const':
        mask = (mask > 0)

    if output_path:
        cv2.imwrite(output_path, (mask * 255).astype(np.uint8))

    return mask
    

def map(image=None, image_path=None, format='rgb', map=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], output_path=None):
    """
    Generates a color shifted image for a given image or image path.
    format is the format of the shift map.
    map is the color shifting map, and includes 3 tuples (1 for each channel), each representing the color to be added for that channel.
    """
    image = get_image(image, image_path, format)

    image = image.astype(np.float32)
    h, w, _ = image.shape
    mapped_image = np.zeros((h, w, 3))

    for i in range(3):
        for j in range(3):
            mapped_image[:, :, j] += image[:, :, i] * (map[i][j] / 255.0)

    mapped_image = np.clip(mapped_image, 0, 255)

    if output_path:
        if format == 'rgb':
            cv2.imwrite(output_path, cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(output_path, cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_HSV2BGR))

    if format == 'rgb':
        mapped_image = cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        mapped_image = cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
    return mapped_image

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

def increase_contrast(image, amt=None, pivot=None):
    """
    Takes in an image and scales the difference between each pixel color and the pivot by {amt}.
        If amt is None, no change occurs.
        If pivot is None, average value of the pixels is used.
    Returns the image.
    """
    if amt is None:
        return image
    img = image.astype(np.float32)
    if pivot is None:
        pivot = np.mean(img, axis=(0, 1), keepdims=True)
    img = (img - pivot) * amt + pivot
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def greyscale(image):
    """
    Takes in an image and returns the greyscaled version.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255