import os
from pathlib import Path

import numpy as np

from neo_ascii.ascii_masks import generate_ascii_mask
from neo_ascii.effects import generate_pulsing_mask, generate_rain_mask
from neo_ascii.transforms import map, greyscale

from neo_ascii.scaler import scale
from neo_ascii.utils import ascii_scaled_dims, ascii_scaled, mask_to_image
from neo_ascii.utils import extension_type

from neo_ascii.assembler import assemble
from neo_ascii.exporter import image_to_video, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (80, 45)
input_path = dir_path / 'input.png'
output_path = dir_path / 'output.gif'

def main():

    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]))
    effect_mask = np.clip(generate_pulsing_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), density=1, background=0.6) + generate_rain_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]))*0.12, 0, 1)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    activation_mask = greyscale(image)
    color_mask = map(image=mask_to_image(effect_mask), format='hsv', map=[(0,0,0), (0,0,0), (0,255,255)])
    color_mask = map(image=color_mask, map=[(230,255,230), (-180,255,-180), (-180,255,-180)])
    return assemble(ascii_mask=ascii_mask, color_mask=color_mask, activation_mask=activation_mask, effect_mask=effect_mask)

if __name__ == '__main__':
    main()