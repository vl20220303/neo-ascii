import os
from pathlib import Path

from neo_ascii.ascii_masks import generate_ascii_mask
from neo_ascii.effects import generate_rain_mask
from neo_ascii.transforms import map, brighten, greyscale

from neo_ascii.scaler import scale
from neo_ascii.utils import ascii_scaled_dims, ascii_scaled
from neo_ascii.utils import extension_type

from neo_ascii.assembler import assemble
from neo_ascii.exporter import image_to_image, video_to_video

dir_path = Path(os.path.dirname(os.path.abspath(__file__)))

dimensions = (120, 80)
input_path = dir_path / 'input.png'
output_path = dir_path / 'output.png'

def main():

    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(dimensions[0], dimensions[1]), density=0.99)

    if extension_type(input_path) in {'.jpg', '.png', '.jpeg'}:
        image_to_image(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)

    elif extension_type(input_path) in {'.mp4', '.gif'}:
        video_to_video(input_path, output_path, pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=None)
    

def pipeline(image, ascii_mask, effect_mask):
    image = scale(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    activation_mask = greyscale(map(image=image, map=[(255, 255, 255), (255, 255, 255), (255, 255, 255)]))
    color_mask = map(image=brighten(image), map=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    return assemble(color_mask=color_mask, activation_mask=activation_mask, use_ascii_activation="contrast")

if __name__ == '__main__':
    main()