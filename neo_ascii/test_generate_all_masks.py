import os
from pathlib import Path

import cv2
import numpy as np
import imageio

from neo_ascii.image_mask_generators import (
    get_image,
    generate_ascii_mask,
    generate_rain_mask,
    generate_pulsing_mask,
    generate_threshold_mask,
    generate_color_mask,
)

def main():
    # -- Configuration --------------------------------------------------------------------
    TEST_DIR = Path(__file__).parent / "tests" / "image_mask_test"
    INPUT_FILE = TEST_DIR / "input.png"

    # make sure directories exist
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # if input.png doesn’t exist, create a simple placeholder
    if not INPUT_FILE.exists():
        placeholder = np.zeros((64, 128, 3), dtype=np.uint8)
        placeholder[:32, :64] = (0, 0, 255)   # red block
        placeholder[32:, 64:] = (0, 255, 0)   # green block
        cv2.imwrite(str(INPUT_FILE), placeholder)

    # -- 1) Visualize the raw input in both RGB & HSV ------------------------------------
    OUTPUT_DIR = TEST_DIR / "raw"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb = get_image(None, str(INPUT_FILE), format="rgb")
    hsv = get_image(None, str(INPUT_FILE), format="hsv")
    cv2.imwrite(str(OUTPUT_DIR / "input_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUTPUT_DIR / "input_hsv.png"), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    
    print(f"Output successfully written to {OUTPUT_DIR}")

    # -- 2) Threshold mask (HSV) ---------------------------------------------------------
    OUTPUT_DIR = TEST_DIR / "threshold_mask"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # const activation
    thresh = generate_threshold_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="hsv",
        include=[(100, 20), (100, 255), (90, 255)],
        activation="const",
    )
    thresh_img = (thresh.astype(np.uint8) * 255)
    cv2.imwrite(str(OUTPUT_DIR / "threshold_const.png"), thresh_img)

    # linear activation
    thresh_lin = generate_threshold_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="hsv",
        include=[(30, 80), (50, 255), (50, 255)],
        activation="linear",
    )
    thresh_lin_img = (thresh_lin * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "threshold_linear.png"), thresh_lin_img)

    # 2-sided-linear activation
    thresh_lin = generate_threshold_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="hsv",
        include=[(0, 20), (80, 255), (80, 255)],
        activation="2-sided-linear",
    )
    thresh_lin_img = (thresh_lin * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "threshold_2_sided_linear.png"), thresh_lin_img)

    print(f"Output successfully written to {OUTPUT_DIR}")

    # -- 3) Color remapping (RGB) --------------------------------------------------------
    OUTPUT_DIR = TEST_DIR / "color_map_rgb"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # identity map
    color_map = [(255, 0,   0),
                (0,   255, 0),
                (0,   0,   255)]
    colormap_img = generate_color_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="rgb",
        map=color_map,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "color_map_identity.png"),
        colormap_img.astype(np.uint8),
    )

    # green tint map
    color_map = [(0, 100, 0),
                (0, 100, 0),
                (0, 100, 0)]
    colormap_img = generate_color_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="rgb",
        map=color_map,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "color_map_tint.png"),
        colormap_img.astype(np.uint8),
    )

    # yellow/pink/blue map
    color_map = [(255, 255, 0),
                (255, 108, 180),
                (0, 0, 100)]
    colormap_img = generate_color_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="rgb",
        map=color_map,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "color_map_shift.png"),
        colormap_img.astype(np.uint8),
    )

    print(f"Output successfully written to {OUTPUT_DIR}")

    # -- 3.5) Color remapping (HSV) --------------------------------------------------------
    OUTPUT_DIR = TEST_DIR / "color_map_hsv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # identity map
    color_map = [(255, 0,   0),
                (0,   255, 0),
                (0,   0,   255)]
    colormap_img = generate_color_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="hsv",
        map=color_map,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "color_map_identity_hsv.png"),
        colormap_img.astype(np.uint8),
    )

    # red tint map
    color_map = [(0, 0, 0),
                (0, 0, 0),
                (0, 255, 255)]
    colormap_img = generate_color_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="hsv",
        map=color_map,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "color_map_tint_hsv.png"),
        colormap_img.astype(np.uint8),
    )

    # shift map
    color_map = [(0,  255,  255),
                (30, 255,  255),
                (100, 255, 255)]
    colormap_img = generate_color_mask(
        image=None,
        image_path=str(INPUT_FILE),
        format="hsv",
        map=color_map,
    )
    cv2.imwrite(
        str(OUTPUT_DIR / "color_map_shift_hsv.png"),
        colormap_img.astype(np.uint8),
    )

    print(f"Output successfully written to {OUTPUT_DIR}")

    # -- 4) Rain mask --------------------------------------------------------------------
    OUTPUT_DIR = TEST_DIR / "rain_mask"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rain = generate_rain_mask(
        image_dims=None,
        image_path=str(INPUT_FILE),
        density=0.4,
        drop_height=0.7,
        cycle=None
    )

    # Save frame 0
    rain_frame0 = (rain[0] * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "rain_frame0.png"), rain_frame0)

    # Save average over time
    rain_avg = (rain.mean(axis=0) * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "rain_average.png"), rain_avg)

    # -- Create GIF from rain frames ----------------------------------------------------
    frames = [(frame * 255).astype(np.uint8) for frame in rain]

    # Save GIF
    gif_path = OUTPUT_DIR / "rain.gif"
    imageio.mimsave(gif_path, frames, format='GIF', duration=0.0005)  # ~160 fps

    print(f"Output successfully written to {OUTPUT_DIR}")

    # -- 5) Pulsing mask ---------------------------------------------------------------
    OUTPUT_DIR = TEST_DIR / "pulse_mask"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # pulse
    pulse = generate_pulsing_mask(
        image_dims=None,
        image_path=str(INPUT_FILE),
        density=0.8,
        effect_type="pulse",
        cycle=None
    )
    
    pulse_t0 = (pulse[:, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "pulse_t0.png"), pulse_t0)

    pulse_avg = (pulse.mean(axis=2) * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "pulse_average.png"), pulse_avg)

    # -- Create GIF from rain frames ----------------------------------------------------
    frames = [(frame * 255).astype(np.uint8) for frame in pulse]

    # Save GIF
    gif_path = OUTPUT_DIR / "pulse.gif"
    imageio.mimsave(gif_path, frames, format='GIF', duration=0.001)  # ~80 fps

    # raindrop
    pulse = generate_pulsing_mask(
        image_dims=None,
        image_path=str(INPUT_FILE),
        density=0.9,
        effect_type="raindrop",
        cycle=None
    )

    pulse_t0 = (pulse[:, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "raindrop_t0.png"), pulse_t0)

    pulse_avg = (pulse.mean(axis=2) * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "raindrop_average.png"), pulse_avg)

    # -- Create GIF from rain frames ----------------------------------------------------
    frames = [(frame * 255).astype(np.uint8) for frame in pulse]

    # Save GIF
    gif_path = OUTPUT_DIR / "raindrop.gif"
    imageio.mimsave(gif_path, frames, format='GIF', duration=0.001)  # ~80 fps

    # beacon
    pulse = generate_pulsing_mask(
        image_dims=None,
        image_path=str(INPUT_FILE),
        density=0.9,
        effect_type="beacon",
        cycle=None
    )

    pulse_t0 = (pulse[:, :, 0] * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "beacon.png"), pulse_t0)

    pulse_avg = (pulse.mean(axis=2) * 255).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "beacon_average.png"), pulse_avg)

    # -- Create GIF from rain frames ----------------------------------------------------
    frames = [(frame * 255).astype(np.uint8) for frame in pulse]

    # Save GIF
    gif_path = OUTPUT_DIR / "beacon.gif"
    imageio.mimsave(gif_path, frames, format='GIF', duration=0.001)  # ~80 fps

    print(f"Output successfully written to {OUTPUT_DIR}")

    # -- 6) ASCII mask as grayscale visualization --------------------------------------
    OUTPUT_DIR = TEST_DIR / "ascii_mask"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ascii_mask = generate_ascii_mask(image_dims=None, image_path=str(INPUT_FILE))

    # Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    char_spacing = 12  # pixels between characters
    line_spacing = 14  # pixels between lines

    height, width = ascii_mask.shape
    img_height = height * line_spacing
    img_width = width * char_spacing

    canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8) * 255

    for i in range(height):
        for j in range(width):
            char = ascii_mask[i, j]
            x = j * char_spacing
            y = (i + 1) * line_spacing
            cv2.putText(canvas, char, (x, y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    cv2.imwrite(str(OUTPUT_DIR / "ascii_mask_rendered.png"), canvas)

    print(f"Output successfully written to {OUTPUT_DIR}")

    
if __name__ == '__main__':
    main()