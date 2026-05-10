import numpy as np

from neo_ascii.utils import get_dims

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