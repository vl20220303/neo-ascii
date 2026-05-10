import cv2
import numpy as np

from neo_ascii.utils import get_dims, get_image

def generate_rain_mask(image_dims=None, image=None, image_path=None, density=0.5, drop_height=0.8, cycle=None):
    """
    Generates a raining effect mask for a given image dimension, image, or image path.
    density is the rough proportion of the screen that will be covered at a single moment.
    drop_height is the height of the raindrop.
    cycle is the number of frames to be generated, which is expected to be larger than the image height to create a loopable animation.
        If None is provided, the image height will be used.
    """
   
    height, width = get_dims(image_dims, image, image_path)

    if cycle is None:
        cycle = 2*height
    elif cycle <= height:
        raise ValueError("Cycle length must be greater than screen height!")
    
    drop = np.ones(int(height*drop_height))
    drop_tail_height = int(height*drop_height*0.2)
    drop_tail = np.linspace(0, 1 - 1/drop_tail_height, drop_tail_height)
    drop[:drop_tail_height] = drop_tail
    drop_len = len(drop)
    
    repeat_length = int(1.5*height)

    mask = np.zeros((cycle+repeat_length, width))

    num_drops = int (cycle * density)
    drop_coords = [(np.random.randint(height+drop_len, cycle+repeat_length), np.random.randint(0, width)) for _ in range(num_drops)]

    for x, y in drop_coords:
        if x - drop_len >= height:
            mask[x - drop_len:x, y] += drop

    repeat = mask[-repeat_length:, :]
    mask[:repeat_length, :] += repeat

    np.clip(mask, 0, 1, out=mask)

    final_mask = np.zeros((cycle, height, width))
    for i in range(cycle):
        start = cycle - i
        end = start + height
        final_mask[i] = mask[start:end, :]
        
    return final_mask

    
def generate_pulsing_mask(image_dims=None, image = None, image_path=None, density=0.5, effect_type='pulse', cycle=None, background=0.3):
    """
    Generates a pulsing effect mask for a given image dimension, image, or image path.
    density is the rough proportion of the screen that will be covered at a single moment.
    effect_type is the name of the effect to be applied.
        Includes 'pulse', 'raindrop', 'beacon'.
    cycle is the number of frames to be generated, which is expected to be larger than the image height to create a loopable animation.
        If None is provided, the image height will be used.
    background is the background brightness of the pixels.
    """

    height, width = get_dims(image_dims, image, image_path)

    if cycle is None:
        cycle = 2*height
    elif cycle <= height:
        raise ValueError("Cycle length must be greater than screen height!")
    
    if effect_type not in {'pulse', 'raindrop', 'beacon'}:
        raise ValueError("Effect must be one of 'pulse', 'raindrop', or 'beacon'!")
    
    num_drops_col = int (cycle * density)
    drop_coords_temporal = []
    for _ in range(num_drops_col):
        h = np.random.randint(0, height)
        w = np.random.randint(0, width)
        size = np.random.randint(1, height / 4)
        time = np.random.randint(0, int(cycle - size))
        drop_coords_temporal.append((h, w, size, time))

    mask = np.zeros((width, cycle, height))
    mask.fill(background)

    for h, w, size, time in drop_coords_temporal:
        pulse_min = np.array([(max(0, 1 - (i - size/2)**2))**0.5 for i in range(size)])
        
        subsize = size
        if effect_type == 'pulse':
            subsize*=0.5
        elif effect_type == 'raindrop':
            subsize*=0.1
        elif effect_type == 'beacon':
            subsize*=0.8
        subsize = int(subsize)

        pulse_increase = np.append(np.linspace(0, 1, subsize), np.linspace(1, 0, size-subsize))
        pulse = pulse_min + pulse_increase[:, np.newaxis]
        h_start = max(0, h - size // 2)
        h_end = min(mask.shape[2], h + size // 2)
        t_start = max(0, time)
        t_end = min(mask.shape[1], time + size)

        pulse_cropped = pulse[:t_end - t_start, :h_end - h_start]

        mask[w, t_start:t_end, h_start:h_end] += pulse_cropped

    np.clip(mask, 0, 1, out=mask)

    final_mask = mask.transpose(1, 2, 0)

    return final_mask