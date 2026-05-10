import os
from pathlib import Path

from neo_ascii.transforms import (
    map,
    brighten, 
    greyscale
)

from neo_ascii.scaler import scale
from neo_ascii.utils import ascii_scaled
from neo_ascii.utils import extension_type

from neo_ascii.assembler import assemble
from neo_ascii.exporter import image_to_image, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (80, 80)
input_path = dir_path / 'input.png'
output_path = dir_path / 'output.png'

def main():
    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_image(input_path, output_path, pipeline=pipeline, ascii_mask=None, effect_mask=None)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=None, effect_mask=None)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    activation_mask = greyscale(image)
    color_mask = map(image=brighten(image), map=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    activation_color = (255, 200, 100)
    return assemble(color_mask=color_mask, activation_mask=activation_mask, use_ascii_activation="block", activation_color=activation_color)

if __name__ == '__main__':
    main()