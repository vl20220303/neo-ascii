import cv2
import numpy as np
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
from neo_ascii.image_helpers import mask_to_image, ascii_scaled, oversaturate

def assemble_masks(ascii_mask=None, color_mask=None, effect_mask=None, activation_mask=None, output_path=None, params=None, use_ascii_activation=None):

    # defaults
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    char_spacing = 12
    line_spacing = 14

    # ascii brightness
    ascii_activation = {
        'default'    : ['@', '%', '#', '*', '+', '=', '-', ':', '.', ' '],
        'block'      : ['█', '▓', '▒', '░', '#', '*', '+', '-', '.', ' '],
        'minimalist' : ['#', 'A', 'X', 'x', '+', '=', ':', '.', '-', ' '],
        'contrast'   : ['M', 'N', 'H', '#', 'Q', 'U', 'A', 'T', '.', ' ']
    }

    if not params is None:
        font, font_scale, thickness, char_spacing, line_spacing = params

    for mask in [ascii_mask, color_mask, effect_mask, activation_mask]:
        if mask is not None:
            height, width = mask.shape[:2]
            break
    else:
        raise ValueError("No valid mask provided to determine shape.")
    
    img_height = height * line_spacing
    img_width = width * char_spacing

    canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8) * 255

    for i in range(height):
        for j in range(width):


            color = color_mask[i, j] if color_mask is not None else np.array((255, 255, 255))
            effect = effect_mask[i, j] if effect_mask is not None else 1
            activation = activation_mask[i, j] if activation_mask is not None else 1
            
            color = color.astype(float)
            color*=effect*activation
            color = tuple(int(c) for c in color)

            if ascii_mask is not None:
                char = ascii_mask[i, j] 
            elif use_ascii_activation is not None:
                final_activation = np.sum(color) / (3*255)
                char = ascii_activation.get(use_ascii_activation, ascii_activation.get('default'))[min(int(final_activation*10), 9)]
            else:
                char = '█'

            x = j * char_spacing
            y = (i + 1) * line_spacing  # OpenCV anchors text at baseline
            cv2.putText(canvas, char, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    if output_path is not None:
        cv2.imwrite(output_path, canvas)
    
    return canvas

def main():
    dir_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'tests/image_assembler_test'
    input_path = dir_path / 'input.png'
    output_path = dir_path / 'output.png'
    demo_pipeline(input_path, output_path, dir_path / 'debug')


def demo_pipeline(input_path, output_path, debug_dir):
    image = cv2.imread(input_path)
    image = scale_image(image=image, new_size=(150, 150))
    image = ascii_scaled(image)
    ascii_mask = generate_ascii_mask(image=image)
    effect_mask = generate_rain_mask(image=image, density=1)[0]
    activation_mask = generate_threshold_mask(
        image=image,
        format="hsv",
        include=[(0, 180), (10, 255), (50, 255)],
        activation="linear",
        output_path=debug_dir / 'activation.png'
    )
    color_mask = generate_color_mask(image=image, map=[(0,255,0), (0,255,0), (0,255,0)], output_path=debug_dir / 'color.png')
    assemble_masks(ascii_mask=ascii_mask, color_mask=color_mask, effect_mask=effect_mask, output_path=output_path)


if __name__ == '__main__':
    main()