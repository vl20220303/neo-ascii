import os
from pathlib import Path

import numpy as np

from neo_ascii.image_mask_generators import (
    generate_ascii_mask,
    generate_rain_mask,
    generate_pulsing_mask,
    generate_threshold_mask,
    generate_color_mask
)
from neo_ascii.image_scaler import scale_image
from neo_ascii.image_helpers import ascii_scaled_dims, mask_to_image, ascii_scaled, oversaturate, brighten, greyscale
from neo_ascii.image_helpers import extension_type

from neo_ascii.image_assembler import assemble_masks
from neo_ascii.exporter import image_to_image, image_to_video, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (180, 180)
input_path = dir_path / 'input.png'
output_path = dir_path / 'output.mp4'

def main():

    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), drop_height=0.3)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask, duration=1/150)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask, duration=1/150)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale_image(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    activation_mask = np.clip(greyscale(generate_color_mask(image=image, map=[(255, 55, 55), (55, 255, 55), (55, 55, 255)])) + effect_mask, 0, 1)
    color_mask = generate_color_mask(image=oversaturate(brighten(image, 100), 60), map=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    color_mask = np.clip(color_mask.astype(float) + generate_color_mask(image=mask_to_image(effect_mask), map=[(0, 0, 0), (0, 0, 0), (120, 120, 220)]).astype(float), 0, 255).astype(np.uint8)
    return assemble_masks(color_mask=color_mask, activation_mask=activation_mask, use_ascii_activation="contrast")

if __name__ == '__main__':
    main()