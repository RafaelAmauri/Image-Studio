import numpy as np
import PIL.Image

import include.effects.edge_detection as edge_detection
import include.effects.colormapping as colormapping
import include.effects.brightness as brightness
import include.effects.contrast as contrast
import include.effects.quantize as quantize
import include.effects.dither as dither
import include.effects.blur as blur

import include.utils.colormodel as colormodel
import include.utils.parser as parser


def main(args):

    # Open the image
    img = PIL.Image.open(args.image)
    img = np.asarray(img, dtype=np.uint8)

    # Convert to grayscale if so desired. The change back to RGB is to add a 3-channel dimension to the image.
    # This simplifies the integration with the rest of the code.
    if args.grayscale:
        img = colormodel.rgb2grayscale(img)
    
    

    # Creates a uniformily spaced color distribution. It's a uniform division from 0 to 255, with args.quantize different colors.
    availableColors = np.linspace(0, 255, args.quantize, dtype=np.uint8)

    if args.contrast != -1:
        img = contrast.contrast_boost(img, args.contrast)

    if args.brightness != -256:
        img = brightness.brightness_boost(img, args.brightness)

    # If the user wants to quantize the image. args.quantize contains the number of colors available. If args.quantize is 255 (the default value),
    # then there's no need to apply quantization.
    if args.quantize != 255:
        # Quantize the image with dithering
        if args.dithering is not None:
            if args.dithering == "ordered":
                img = dither.orderedDithering(img, 2, availableColors)
            elif args.dithering == "floyd-steinberg":
                img = dither.floydSteinberg(img, availableColors)

        # Quantize the image without dithering
        if args.dithering is None:
            img = quantize.quantize(img, availableColors)

    
    # Change the color palette acording to a user-specified hue
    if args.hue is not None:
        # This is kinda crazy, but we have to use separate functions depending if the image is Grayscale or if it is RGB.
        # That's because if the image is in grayscale, then the available colors are... well... the array availableColors.

        # But if the image is RGB, then the available colors are all the unique combinations in the R, G and B channel.
        # That's because even though we quantize the image with an arbitrary number of colors, that reduced number of
        # colors can COMBINE INTO DIFFERENT colors because of the 3 channels. For example, if there's only 3 colors for each channel:
        # [0, 127, 255], then there's 3 * 3 * 3 different combinations of colors.
        # This is what ends up giving us a very large number of different Hues, and the reason why
        # the colors available in the RGB image are the unique values in hsvImg[..., 0] instead of availableColors :)
        if args.grayscale:
            # Since we just have an rgb2hsv function and not a grayscale2hsv function, we have to repeat the channel dimension 3 times
            # to make the grayscale image work as an RGB image.
            img      = np.repeat(img, repeats=3, axis=2)
            hsvImg   = colormodel.rgb2hsv(img)
            colorLUT = colormapping.generatePalette(args.hue, availableColors, args.hue_range, args.hue_reversed)
            hsvImg   = colormapping.changeColorPaletteGrayscale(hsvImg, colorLUT)
        else:
            hsvImg   = colormodel.rgb2hsv(img)
            colorLUT = colormapping.generatePalette(args.hue, np.unique(hsvImg[..., 0]), args.hue_range, args.hue_reversed)
            hsvImg   = colormapping.changeColorPaletteRGB(hsvImg, colorLUT)
            

        img  = colormodel.hsv2rgb(hsvImg)


    if args.blur is not None:
        # Perform image blur
        img = blur.blur(img, args.blur)


    if args.edge_detection is not None:
        # Perform sobel edge detection
        if args.edge_detection == "sobel":
            img = edge_detection.sobel(img, args.edge_color)
        # Perform prewitt edge detection
        if args.edge_detection == "prewitt":
            img = edge_detection.prewitt(img, args.edge_color)


    if img.shape[-1] == 1:
        # Remove the fake channel dimension
        img = img.squeeze(axis=2)


    # Save the image
    img = PIL.Image.fromarray(img)
    img.save("./processed.png")


if __name__ == '__main__':
    args = parser.make_parser().parse_args()
    parser.validateParams(args)

    main(args)