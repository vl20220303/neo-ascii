import os
from pathlib import Path

import numpy as np

from neo_ascii.image_mask_generators import (
    generate_ascii_mask,
    generate_rain_mask,
    generate_color_mask,
    generate_threshold_mask
)
from neo_ascii.image_scaler import scale_image
from neo_ascii.image_helpers import ascii_scaled_dims, ascii_scaled, brighten, greyscale, mask_to_image
from neo_ascii.image_helpers import extension_type

from neo_ascii.image_assembler import assemble_masks
from neo_ascii.exporter import image_to_image, image_to_video, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (100, 180)
input_path = dir_path / 'input.png'
output_path = dir_path / 'output.gif'

def main():

    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), density=4)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)


def pipeline(image, ascii_mask, effect_mask):
    image = scale_image(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    activation_mask = generate_threshold_mask(image=image, include=[(0, 180), (10, 255), (50, 255)])
    red_pill = generate_color_mask(image=mask_to_image(generate_threshold_mask(image=image, include=[(170, 8), (150, 255), (180, 255)])), map=[(255, 30, 30), (0, 0, 0), (0, 0, 0)])
    blue_pill = generate_color_mask(image=mask_to_image(generate_threshold_mask(image=image, include=[(80, 110), (60, 255), (80, 255)])), map=[(0, 100, 255), (0, 0, 0), (0, 0, 0)])
    pills_mask = generate_threshold_mask(image=np.clip(red_pill.astype(int) + blue_pill.astype(int), 0, 255).astype(np.uint8), include=[(0, 180), (0, 255), (1, 255)])
    hands = np.clip(mask_to_image(generate_threshold_mask(image=mask_to_image(greyscale(image)), include=[(0, 180), (0, 10), (180, 255)]))*0.7 + 
                    mask_to_image(generate_threshold_mask(image=mask_to_image(greyscale(image)), include=[(0, 180), (0, 10), (90, 255)]))*0.3 -
                    mask_to_image(pills_mask), 0, 255).astype(np.uint8)
    hands = generate_color_mask(image=hands, map=[(255, 255, 255), (0, 0, 0), (0, 0, 0)])
    color_mask = np.clip(red_pill.astype(int) + blue_pill.astype(int) + hands, 0, 255).astype(np.uint8)
    effect_mask = np.clip(effect_mask + pills_mask, 0, 1)
    return assemble_masks(ascii_mask=ascii_mask, color_mask=color_mask, activation_mask=activation_mask, effect_mask=effect_mask)

if __name__ == '__main__':
    main()