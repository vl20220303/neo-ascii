import os
from pathlib import Path

from neo_ascii.ascii_masks import generate_ascii_mask
from neo_ascii.effects import generate_rain_mask
from neo_ascii.transforms import map

from neo_ascii.scaler import scale
from neo_ascii.utils import ascii_scaled_dims, ascii_scaled, extension_type

from neo_ascii.assembler import assemble
from neo_ascii.exporter import image_to_video, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (90, 90)
input_path = dir_path / 'input.png'
output_path = dir_path / 'output.gif'

def main():

    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), density=0.9)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask, duration=0.05)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    color_mask = map(image=image, map=[(0,255,0), (0,255,0), (0,255,0)])
    return assemble(ascii_mask=ascii_mask, color_mask=color_mask, effect_mask=effect_mask)

if __name__ == '__main__':
    main()