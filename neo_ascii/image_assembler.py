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
from neo_ascii.image_helpers import mask_to_image, ascii_scaled, oversaturate, brighten

def assemble_masks(ascii_mask=None, color_mask=None, effect_mask=None, activation_mask=None, output_path=None, params=None, use_ascii_activation=None, activation_color=None):
    """
    ascii_mask is expected to be a single mask (2D).
        If None is provided, brightness characters are used (if provided).
    color_mask is expected to be a single mask (2D).
        If None is provided, white color is used.
    effect_mask is expected to be a single mask (2D).
        If None is provided, no effect is applied.
    activation_mask is expected to be a single mask (2D).
        If None is provided, an activation of 1 (no reduction) is used.
    params is the character params, and should include all of (font, font_scale, thickness, char_spacing, line_spacing)
        If None is provided, defaults (HERSHEY_SIMPLEX, 0.4, 1, 12, 14) are used.
    use_ascii_activation is the brightness characters, when ascii_mask is None and an activation type is provided.
        Includes 'default', 'block', 'minimalist', 'contrast'.
        If None is provided, and ascii_mask is not provided, pixel blocks are used with color_mask.
    activation_color is the color of the brightness characters, in BGR format.
        If None is provided, the color_mask is used.
    """

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

            orig_color = color_mask[i, j] if color_mask is not None else np.array((255, 255, 255))
            effect = effect_mask[i, j] if effect_mask is not None else 1
            activation = activation_mask[i, j] if activation_mask is not None else 1

            color = orig_color.astype(float)
            color*=effect*activation

            if ascii_mask is not None:
                char = ascii_mask[i, j]
            elif use_ascii_activation is not None:
                final_activation = np.sum(color) / (3*255)
                char = ascii_activation.get(use_ascii_activation, ascii_activation.get('default'))[9 - min(int(final_activation*10), 9)]
                if activation_color is not None:
                    color = activation_color
                else:
                    color = orig_color
            else:
                char = '█'
                
            color = tuple(int(c) for c in color)

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