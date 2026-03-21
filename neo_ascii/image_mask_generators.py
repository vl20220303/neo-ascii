import cv2
import numpy as np

from neo_ascii.image_helpers import get_dims, get_image

def generate_ascii_mask(image_dims=None, image=None, image_path=None, chars=None):
    """
    Generates an array of characters for a given image dimension, image, or image path.
    chars is the character array to choose from (at random).
        If None is provided, the default ($,#,&,%,+,-,0-10,a-z,A-Z) is used.
    """
    if chars is None:
        chars = []
        chars += ['$', '#', '&', '%', '+', '-'] * 10
        chars += [str(i) for i in range(0, 10)] * 10
        chars += [chr(i) for i in range(ord('a'), ord('z'))] 
        chars += [chr(i) for i in range(ord('A'), ord('Z'))]

    dims = get_dims(image_dims, image, image_path)
    
    mask = np.random.default_rng().choice(chars, size=dims, replace=True)
    return mask

def generate_static_phrase_mask(image_dims=None, image=None, image_path=None, phrases=['ASCII'], direction=1):
    """
    Generates an array of phrases for a given image dimension, image, or image path.
    phrases is the phrase array to choose from (at random).
    direction is the direction the phrases are read (1=left-to-right, 2=top-to-bottom, 3=right-to-left, 4=bottom-to-top)
    """

    height, width = get_dims(image_dims, image, image_path)
    
    char_phrases = []
    for phrase in phrases:
        phrase_chars = np.array(list(phrase), dtype='<U1')
        char_phrases.append(phrase_chars)

    lines = []
    if direction%2==1:
        for _ in range(height):
            temp = []
            while len(temp) < width:
                temp.extend(char_phrases[np.random.randint(len(char_phrases))])
            temp = temp[:width]
            temp = np.roll(np.array(temp, dtype='<U1'), np.random.randint(width))
            lines.append(temp)
        
        mask = np.array(lines)

    else:
        for _ in range(width):
            temp = []
            while len(temp) < height:
                temp.extend(char_phrases[np.random.randint(len(char_phrases))])
            temp = temp[:height]
            temp = np.roll(np.array(temp, dtype='<U1'), np.random.randint(height))
            lines.append(temp)
        
        mask = np.array(lines).T
    
    if direction > 2:
        mask = mask[:, ::-1]
    
    return mask

def generate_vertical_phrase_mask(image_dims=None, image=None, image_path=None, phrases=['ASCII'], scroll_direction=1, speed=0.3, cycle=None):
    """
    Generates scrolling phrases for a given image dimension, image, or image path.
    phrases is the phrase array to choose from (at random).
    scroll_direction is the probability phrases move upwards or downwards (0=purely upwards, 1=purely downwards). Phrases will always read top-down.
    cycle is the number of frames to be generated, which is expected to be larger than the image height to create a loopable animation.
    """

    height, width = get_dims(image_dims, image, image_path)

    if cycle is None:
        cycle = 2*height
    elif cycle <= height:
        raise ValueError("Cycle length must be greater than screen height!")
    
    char_phrases = []
    for phrase in phrases:
        phrase_chars = np.array(list(phrase), dtype='<U1')
        char_phrases.append(phrase_chars)

    lines = []; lines_dir = []
    for i in range(width):
        temp = []
        while len(temp) < height+cycle:
            temp.extend(char_phrases[np.random.randint(len(char_phrases))])
        temp = temp[:height+cycle]
        temp = np.roll(np.array(temp, dtype='<U1'), np.random.randint(height+cycle))
        lines.append(temp)
        lines_dir.append(np.random.random()<=scroll_direction)

    final_mask = np.empty((cycle, height, width), dtype='<U1')
    for i in range(min(cycle, int(cycle*speed))):
        mask = []
        for j in range(width):
            start = i if lines_dir[j] else cycle - i
            end = start+height
            mask.append(lines[j][start:end])
        final_mask[int(i/speed):int((i+1)/speed)] = np.array(mask).T

    return final_mask

def generate_horizontal_phrase_mask(image_dims=None, image=None, image_path=None, phrases=['ASCII'], scroll_direction=1, speed=0.3, cycle=None):
    """
    Generates scrolling phrases for a given image dimension, image, or image path.
    phrases is the phrase array to choose from (at random).
    scroll_direction is the probability phrases move left or right (0=purely left, 1=purely right). Phrases will always read top-down.
    cycle is the number of frames to be generated, which is expected to be larger than the image height to create a loopable animation.
    """

    height, width = get_dims(image_dims, image, image_path)

    if cycle is None:
        cycle = 2*height
    elif cycle <= height:
        raise ValueError("Cycle length must be greater than screen height!")
    
    char_phrases = []
    for phrase in phrases:
        phrase_chars = np.array(list(phrase), dtype='<U1')
        char_phrases.append(phrase_chars)

    lines = []; lines_dir = []
    for i in range(height):
        temp = []
        while len(temp) < width+cycle:
            temp.extend(char_phrases[np.random.randint(len(char_phrases))])
        temp = temp[:width+cycle]
        temp = np.roll(np.array(temp, dtype='<U1'), np.random.randint(width+cycle))
        lines.append(temp)
        lines_dir.append(np.random.random()<=scroll_direction)

    final_mask = np.empty((cycle, height, width), dtype='<U1')
    for i in range(min(cycle, int(cycle*speed))):
        mask = []
        for j in range(height):
            start = cycle - i if lines_dir[j] else i
            end = start+width
            mask.append(lines[j][start:end])
        final_mask[int(i/speed):int((i+1)/speed)] = np.array(mask)

    return final_mask

    
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
    

def generate_threshold_mask(image=None, image_path=None, format='hsv', include=[(0, 180), (0, 255), (0, 255)], activation='const', output_path=None):
    """
    Generates a threshold mask for a given image or image path.
        Pixels must conform to one of 3 criteria to remain.
    format is the format of the criteria.
    include is the selection criteria, and includes 3 tuples (1 for each channel), each containing an upper and lower bound that wraps around.
    activation is the activation method of the mask.
        Includes 'const', 'linear', '2-sided-linear'
    """
    image = get_image(image, image_path, format)

    if activation not in {'const', 'linear', '2-sided-linear'}:
        raise ValueError("Activation must be 'const', 'linear', or '2-sided-linear'")

    mask = np.ones(image.shape[:2])

    for i, (low, high) in enumerate(include):
        channel = image[:, :, i]
        if low <= high:
            valid = (channel >= low) & (channel <= high)
        else:
            valid = (channel >= low) | (channel <= high)

        mask *= valid

        if activation == 'linear':
            norm = (channel - low) / (high - low)
            norm = np.clip(norm*2, 0.5, 1)
            mask *= norm

        elif activation == '2-sided-linear':
            norm = (channel - low - (high-low)/2) / (high - low)
            mask *= norm

    if activation == 'const':
        mask = (mask > 0)

    if output_path:
        cv2.imwrite(output_path, (mask * 255).astype(np.uint8))

    return mask
    

def generate_color_mask(image=None, image_path=None, format='rgb', map=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], output_path=None):
    """
    Generates a color shifted image for a given image or image path.
    format is the format of the shift map.
    map is the color shifting map, and includes 3 tuples (1 for each channel), each representing the color to be added for that channel.
    """
    image = get_image(image, image_path, format)

    image = image.astype(np.float32)
    h, w, _ = image.shape
    mapped_image = np.zeros((h, w, 3))

    for i in range(3):
        for j in range(3):
            mapped_image[:, :, j] += image[:, :, i] * (map[i][j] / 255.0)

    mapped_image = np.clip(mapped_image, 0, 255)

    if output_path:
        if format == 'rgb':
            cv2.imwrite(output_path, cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(output_path, cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_HSV2BGR))

    if format == 'rgb':
        mapped_image = cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        mapped_image = cv2.cvtColor(mapped_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
    return mapped_image