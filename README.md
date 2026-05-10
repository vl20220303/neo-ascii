# About neo_ascii
`neo_ascii` is a modular library that supports many image effects and converts images, gifs, and videos into ASCII-style media.

# Usage
Typically, there are three main functional blocks involved with producing an image, GIF, or video with `neo_ascii`:

## The preprocessing
Preprocessing involves generating masks that will be used later.

Any animated effect masks will need to be prepared at the very beginning. For example, the output of `effects.generate_rain_mask()` and `effects.generate_pulsing_mask()` should be generated and stored at the beginning.

In addition, any persistent ASCII masks should also be generated at the beginning, including `ascii_masks.generate_ascii_mask()`, `ascii_masks.generate_static_phrase_mask()`, `ascii_masks.generate_vertical_phrase_mask()`, `ascii_masks.generate_horizontal_phrase_mask()`.

**Note:** If ASCII masks are to be dynamically generated, masks will be handled automatically by the assembler and do not require manual generation.
## The pipeline function
The pipeline function is passed to the exporter as a parameter and tells it how to transform a frame into the ASCII-styled output. It must include the parameters `image`, `ascii_mask`, `effect_mask` (in that order) and must return the output of an `assembler.assemble()` call.

Here's a sample implementation of the pipeline function (that can be found in the Agent Smith demo):
```
def pipeline(image, ascii_mask, effect_mask):
    image = scale(image=image, new_size=dimensions)
    image = ascii_scaled(image)
    color_mask = map(image=image, map=[(0,255,0), (0,255,0), (0,255,0)])
    return assemble(ascii_mask=ascii_mask, color_mask=color_mask, effect_mask=effect_mask)
```
that takes in a frame and its corresponding ASCII & effect masks and produces an ASCII-style image with bright green text.
## The exporter
The exporter supports three kinds of exports: `image_to_image()`, `image_to_video()`, and `video_to_video()`. It takes in the input, pipeline, ASCII & effect masks and processes them frame-by-frame to produce an image or video.

Here's a sample call:
```
image_to_video('input.png', 'output.gif', pipeline=pipeline, ascii_mask=ascii_mask, effect_mask=effect_mask, duration=0.05)
```

After running this, the exporter will take some time to assemble the GIF, and *voila!* You have generated colorized ASCII art.
<br><br>
# DOCUMENTATION
[Core Components](#core-components)

[Effects](#effects)

[Utilities](#utilities)
# Core Components
## `assembler`
`assembler.assemble()` assembles individual frame components into a ASCII-style frame. It is supposed to be implemented in the `pipeline()` function implementation, which should automatically provide the `ascii_mask` and `effect_mask`.

That is, there is no need to manually implement the `ascii_mask` and `effect_mask` that feeds into `assemble()`, as they should come directly from the inputs of `pipeline()`:
```
def pipeline(image, ascii_mask, effect_mask):
    image = ascii_scaled(scale(image=image, new_size=OUTPUT_DIMENSIONS))
    return assemble(ascii_mask=ascii_mask, effect_mask=effect_mask, color_mask=image)
```
`color_mask` should be the RGB (3-channel) image, providing the colors that will be used in the final ASCII-style output.

`activation_mask` should be a greyscale (1-channel) image. This mask can be used to selectively dim parts of `color_mask`.

`params` specify what the ASCII characters should look like. It is a tuple with format `(FONT, FONT SCALE, THICKNESS, CHARACTER SPACING, LINE SPACING)`.
 
A default (`params=(HERSHEY_SIMPLEX, 0.4, 1, 12, 14)`) is provided.

The assembler also supports dynamic ASCII mask generation, which is discussed [later](#the-dynamic-ascii-masks).

## `exporter`
Exporter functions are by and large the same; they all require an `input_path` to the input, a valid `output_path`, an `ascii_mask`, an `effect_mask`, and a manually constructed `pipeline` function to put it all together.

`exporter.image_to_image()` converts a given image (`.png`, `.jpeg` supported) into an ASCII-style image (`.png`, `.jpeg` supported). If an effect mask is included in the input, it is expected the effect mask is an animation and the first frame of the effect mask will be used.

`exporter.image_to_video()` converts a given image (`.png`, `.jpeg` supported) into an ASCII-style video (`.gif`, `.mp4` supported). It includes a `duration` paramter that represents the time between frames (it is a framerate setting). MP4s are exported using the AVC1 codec; without it, the export may fail.

`exporter.video_to_video()` converts a given video (`.gif`, `.mp4` supported) into an ASCII-style video (`.gif`, `.mp4` supported). It is functionally the same as `exporter.image_to_video()`, except for the fact that it handles inputs with many frames.

## Effects
### `ascii_masks`
`ascii_masks.generate_ascii_mask()` randomly fills the ASCII mask with `chars`.

A default (`chars=[$,#,&,%,+,-,0-10,a-z,A-Z]`) is provided.

`ascii_masks.generate_static_phrase_mask()` randomly fills the ASCII mask with continuous `phrases`. The `direction` of the phrases is specified: 1=left-to-right, 2=top-to-bottom, 3=right-to-left, 4=bottom-to-top.

`ascii_masks.generate_vertical_phrase_mask()` creates an animated ASCII mask with scrolling `phrases`. The `direction` of the phrases is specified: 0=purely upwards, 1=purely downwards, with values in-between dictating what proportion of the lines move downwards or upwards.

`ascii_masks.generate_horizontal_phrase_mask()` creates an animated ASCII mask with scrolling `phrases`. The `direction` of the phrases is specified: 0=purely left, 1=purely right, with values in-between similarly dictating what proportion of the lines move left or right.

## The dynamic ASCII masks
If an ASCII mask is not provided in `assembler.assemble()`, the assembler will automatically generate a dynamic ASCII mask per-frame based on the brightness values of `color_mask`. There are four different character sets that can be used to represent brightness (specified by `use_ascii_activation`):

```
        'default'    : ['@', '%', '#', '*', '+', '=', '-', ':', '.', ' ']

        'block'      : ['█', '▓', '▒', '░', '#', '*', '+', '-', '.', ' ']

        'minimalist' : ['#', 'A', 'X', 'x', '+', '=', ':', '.', '-', ' ']

        'contrast'   : ['M', 'N', 'H', '#', 'Q', 'U', 'A', 'T', '.', ' ']
```

You can also use an `activation_color` mask alongside these dynamically generated masks to color the ASCII characters a different color than what is provided in `color_mask` (good for hue shifts and inversions).

### `effects`
`effects.generate_rain_mask()` generates a looping code rain effect. The `drop_height` of each droplet is specified as a proportion of the image height. The `density` of droplets is expressed as decimal, where `density` is the rough proportion of the screen that is to be covered by droplets at any given moment.

`effects.generate_pulsing_mask()` generates pulsing lights. Similarly, `density` is the rough proportion of the screen that is to be covered by the pulses at any given moment. There are three different pulse types:
* `pulse`, which activates/deactivates symmetrically.

* `raindrop`, which pulses brightly and has a slow fade-out (like a raindrop's ripples).

* `beacon`, which slowly glows brighter & brighter before rapidly shutting off.

### `transforms`
`transforms.filter()` provides a way to filter for HSV or RGB values. Specify the `format` of the filter and specify the range of values to include in `include`, which is expected to be a 3-tuple list, where each tuple is in the format of `(LOWERBOUND, UPPERBOUND)` for its respective channel. `include` wraps around; for example a selection criteria of `format='rgb', include=[(200,50),(0,255),(0,255)]` will preserve pixels with R(ed) values between 200-255 and 0-50.

`filter()` also features different activation functions:
* `const` outputs a binary mask

* `linear` increases the activation of values closer to the upperbound

* `2-sided-linear` increases the activation of values closer to the lower or upper bounds.<br><br>

`transforms.map()` produces a linearly colormapped version of the input image using either an RGB or HSV map. Specify the `format` of the map and specify the `map` of each channel. `map` is expected to be a 3-tuple list, where each tuple is in the format of `(CHANNEL 1, CHANNEL 2, CHANNEL 3)` that represents a color that the respective channel maps to.

`transforms.oversaturate()` saturates an image by converting the image to HSV colorspace and adding `amt` to the S(aturation) value. If no `amt` is given, the S-channel is set to max for every pixel.

`transforms.brighten()` brightens an image by converting the image to HSV colorspace and adding `amt` to the V(alue) value. If no `amt` is given, the V-channel is set to max for every pixel.

`transforms.increase_contrast()` increases the contrast of an image by amplifying the difference of each pixel with a `pivot`. `pivot` is expected to be a three-value 1D NumPy array that represents an RGB pivot color. If no `pivot` is provided, the average value of the pixels is used.

`transforms.greyscale()` converts an image to greyscale.

# Utilities
## `scaler`
`scaler.scale()` handles the scaling of images for a given image or image path. You can scale an image to given dimensions by setting `fit_type` to one of three values:
* `crop`, which crops the image to a window of given dimensions a proportion `crop_offset` from the left (this typically useful for converting wide frames to more square frames).

* `contain`, which scales the image to take up as much of the window as possible without cropping out any part of the image, leaving behind black bars.

* `cover`, which scales the image to take up the entire window (at the cost of potentially cropping out parts of the image).
## `utils`
`utils.mask_to_image()` converts a greyscale image / binary mask to an RGB image.

`utils.ascii_scaled()` takes in an image and stretches it vertically (with respect to `params`) to account for the aspect ratio of individual characters (in the assembler, each pixel of the image is converted into one character).

`utils.ascii_scaled_dims()` returns the dimensions if it were to be `ascii_scaled` with respect to given `params`.

`utils.extension_type()` returns the extension type of the file provided.

`utils.get_dims()` returns the dimensions of the given image (typically only used internally as a helper function).

`utils.get_image()` returns the dimensions of the given image in either RGB or HSV format as specified (also an internally used helper function).
