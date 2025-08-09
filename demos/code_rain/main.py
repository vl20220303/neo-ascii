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
from neo_ascii.image_helpers import ascii_scaled_dims, mask_to_image, ascii_scaled, oversaturate
from neo_ascii.image_helpers import extension_type

from neo_ascii.image_assembler import assemble_masks
from neo_ascii.exporter import image_to_image, image_to_video, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (90, 90)
input_path = dir_path / 'input.png'
output_path = dir_path / 'output.gif'

def main():

    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(90, 90))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(90, 90), density=0.9)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale_image(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    color_mask = generate_color_mask(image=image, map=[(0,255,0), (0,255,0), (0,255,0)])
    return assemble_masks(ascii_mask=ascii_mask, color_mask=color_mask, effect_mask=effect_mask)

if __name__ == '__main__':
    main()