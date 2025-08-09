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
from neo_ascii.image_helpers import ascii_scaled_dims, mask_to_image, ascii_scaled, oversaturate

from neo_ascii.image_assembler import assemble_masks

import imageio

def image_to_image(input_path, output_path, pipeline, ascii_mask, effect_mask):
    image = cv2.imread(str(input_path))
    output = pipeline(image, ascii_mask, effect_mask[0] if effect_mask is not None else None)
    print("Writing image...")
    cv2.imwrite(str(output_path), output)
    print(f"Finished writing image to {output_path}")

def image_to_video(input_path, output_path, pipeline, ascii_mask, effect_mask):
    image = cv2.imread(str(input_path))
    frames = []
    for single_effect_mask in effect_mask:
        output = pipeline(image, ascii_mask, single_effect_mask)
        frames.append(output)
    print("Writing GIF...")
    imageio.mimsave(str(output_path), frames, format='GIF', duration=0.05)
    print(f"Finished writing GIF to {output_path}")

def video_to_video(input_path, output_path, pipeline, ascii_mask, effect_mask):
    cap = cv2.VideoCapture(str(input_path))
    frames = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        effect = effect_mask[frame_idx%len(effect_mask)]
        ascii = ascii_mask

        output = pipeline(frame, ascii, effect)
        frames.append(output)
        frame_idx += 1

    cap.release()
    print("Writing GIF...")
    imageio.mimsave(str(output_path), frames, format='GIF', duration=0.05)
    print(f"Finished writing GIF to {output_path}")


def main():
    dir_path = Path(os.path.dirname(os.path.abspath(__file__))) / 'tests/pipelines_test'
    dir_path.mkdir(parents=True, exist_ok=True)
    input_path = dir_path / 'input.png'

    output_path = dir_path / 'output_img_img.png'
    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(100, 100))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(100, 100), density=0.9)
    image_to_image(input_path, output_path, pipeline=demo_img_img_pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)

    output_path = dir_path / 'output_img_gif.gif'
    ascii_mask = generate_ascii_mask(image_dims=ascii_scaled_dims(90, 90))
    effect_mask = generate_rain_mask(image_dims=ascii_scaled_dims(90, 90), density=1)
    image_to_video(input_path, output_path, pipeline=demo_img_gif_pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask)

def demo_img_img_pipeline(image, ascii_mask, effect_mask):
    image = scale_image(image=image, new_size=(100, 100))
    image = ascii_scaled(image)
    activation_mask = generate_threshold_mask(
        image=image,
        format="hsv",
        include=[(0, 180), (10, 255), (50, 255)],
        activation="linear"
    )
    color_mask = generate_color_mask(image=mask_to_image(activation_mask), map=[(0,255,0), (0,255,0), (0,255,0)])
    return assemble_masks(ascii_mask=ascii_mask, color_mask=color_mask, effect_mask=effect_mask)
    

def demo_img_gif_pipeline(image, ascii_mask, effect_mask):
    image = scale_image(image=image, new_size=(90, 90))
    image = ascii_scaled(image)
    color_mask = generate_color_mask(image=image, map=[(0,255,0), (0,255,0), (0,255,0)])
    return assemble_masks(ascii_mask=ascii_mask, color_mask=color_mask, effect_mask=effect_mask)

if __name__ == '__main__':
    main()