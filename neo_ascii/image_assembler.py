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
    activation_color is the color of the brightness characters, in RGB format.
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

    # handling missing color/effect/activations
    color_mask = color_mask if color_mask is not None else np.full((height, width, 3), np.array([255, 255, 255]), dtype=np.uint8)
    effect_mask = effect_mask if effect_mask is not None else np.ones((height, width), dtype=np.float64)
    activation_mask = activation_mask if activation_mask is not None else np.ones((height, width), dtype=np.float64)

    # layering masks together (vectorized)
    full_mask = color_mask.astype(np.float64)
    full_mask *= effect_mask[:, :, np.newaxis]
    full_mask *= activation_mask[:, :, np.newaxis]
    full_mask = np.clip(full_mask, 0, 255).astype(np.uint8)

    # handling ascii (vectorized)
    if ascii_mask is None:
        if use_ascii_activation is not None:
            scale_factor = 10 / (3 * 255)
            activation_values = scale_factor * np.sum(full_mask, axis=2)
            activation_indices = 9 - np.clip(activation_values.astype(int), 0, 9)
            
            ascii_arr = ascii_activation.get(use_ascii_activation, ascii_activation.get('default'))
            ascii_mask = np.array([ascii_arr[i] for i in activation_indices.flatten()]).reshape(height, width)
            if activation_color is not None:
                full_mask[:, :] = activation_color[::-1]
            else:
                full_mask = color_mask
        else:
            ascii_mask = np.full((height, width), '█', dtype='U20')

    for i in range(height):
        for j in range(width):

            color = full_mask[i, j]
            char = ascii_mask[i, j]
                
            color = tuple(int(c) for c in color)

            x = j * char_spacing
            y = (i + 1) * line_spacing  # OpenCV anchors text at baseline
            cv2.putText(canvas, char, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    if output_path is not None:
        cv2.imwrite(output_path, canvas)
    
    return canvas