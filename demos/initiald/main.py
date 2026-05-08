import os
from pathlib import Path

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

dimensions = (60, 120)
input_path = dir_path / 'input.gif'
output_path = dir_path / 'output.gif'

def main():

    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), density=0.99)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_image(input_path, output_path, pipeline=pipeline, ascii_mask=None, effect_mask=None)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=None, effect_mask=None)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale_image(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    activation_mask = greyscale(generate_color_mask(image=image, map=[(255, 255, 255), (255, 255, 255), (255, 255, 255)]))
    color_mask = generate_color_mask(image=oversaturate(brighten(image, 100), 50), map=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    return assemble_masks(color_mask=color_mask, activation_mask=activation_mask, use_ascii_activation="default")

if __name__ == '__main__':
    main()