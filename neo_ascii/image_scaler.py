import cv2
import numpy as np
import os
from pathlib import Path

def scale_image(image=None, image_path=None, new_size=(100, 100), fit_type='crop', crop_offset=0, output_path=None, gen_output_path = True):
    """
    Generates a scaled image for a given image or image path.
    new_size is the new dimensions of the image.
    fit_type is the fitment type
        Includes 'crop', 'contain', 'cover'
    crop_offset is the offset from left used in crop mode
    gen_output_path automatically creates an output file in the same directory as the input file.
    """
    if image is None:
        image = cv2.imread(image_path)
        if image is None:
            print('\033[31mImage not found.')
            return
    
    if output_path is None and gen_output_path and image_path:
        output_path = Path(os.path.dirname(os.path.abspath(image_path))) /  f'output_{new_size}_{str(crop_offset)+'_' if fit_type=='crop' else '_'}{fit_type}.jpg'

    orig_height, orig_width = image.shape[:2]
    new_height, new_width = new_size

    if fit_type == 'crop':
        orig_aspect = orig_width / orig_height
        new_aspect = new_width / new_height

        if new_aspect > orig_aspect:
            crop_height = int(orig_width / new_aspect)
            crop_offset = int(orig_height*crop_offset)
            image = image[crop_offset:crop_offset+crop_height, :]
        else:
            crop_width = int(orig_height * new_aspect)
            crop_offset = int(orig_width*crop_offset)
            image = image[:, crop_offset:crop_offset+crop_width]

    elif fit_type == 'contain':
        scale = min(new_width / orig_width, new_height / orig_height)
        resized = cv2.resize(image, (int(orig_width * scale), int(orig_height * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        y_offset = (new_height - resized.shape[0]) // 2
        x_offset = (new_width - resized.shape[1]) // 2
        new_image[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
        image = new_image

    elif fit_type == 'cover':
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if new_width < orig_width or new_height < orig_height else cv2.INTER_LINEAR)

    else:
        raise ValueError(f"fit_type must be one of 'crop', 'contain', 'cover'")

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if new_width < orig_width or new_height < orig_height else cv2.INTER_LINEAR)
    
    if output_path:
        cv2.imwrite(output_path, image)
    
    return image