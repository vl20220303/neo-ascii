import os
from pathlib import Path

import numpy as np

from neo_ascii.image_mask_generators import (
    generate_ascii_mask,
    generate_rain_mask,
    generate_pulsing_mask,
    generate_static_phrase_mask,
    generate_threshold_mask,
    generate_color_mask
)
from neo_ascii.image_scaler import scale_image
from neo_ascii.image_helpers import ascii_scaled_dims, mask_to_image, ascii_scaled, oversaturate, brighten, greyscale
from neo_ascii.image_helpers import extension_type

from neo_ascii.image_assembler import assemble_masks
from neo_ascii.exporter import image_to_image, image_to_video, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (80, 107)
input_path = dir_path / 'input.mp4'
output_path = dir_path / 'output.mp4'

def main():

    ascii_mask = generate_static_phrase_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), phrases=['badapple'])
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), density=1.3)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_image(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=None)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask, duration=1/30)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale_image(image=image, new_size=dimensions, crop_offset=0.12)
    image = ascii_scaled(image)
    activation_mask = np.clip(generate_threshold_mask(image=image, include=[(0, 180), (0, 255), (100, 255)]) + effect_mask*0.1, 0, 1)
    effect_mask = np.clip(effect_mask + greyscale(image)*0.5, 0, 1)
    return assemble_masks(ascii_mask=ascii_mask, effect_mask=effect_mask, activation_mask=activation_mask)

if __name__ == '__main__':
    main()