import numpy as np
import PIL.Image

import include.colormapping as colormapping
import include.convolve2d as convolve2d
import include.colormodel as colormodel
import include.quantize as quantize
import include.kernels as kernels
import include.dither as dither
import include.parser as parser


def main(args):

    # Open the image
    img = PIL.Image.open(args.image)
    img2 = np.asarray(img.convert("L")).astype(np.uint8)

    img = np.asarray(img, dtype=np.uint8)

    # Convert to grayscale if so desired. The change back to RGB is to add a 3-channel dimension to the image.
    # This simplifies the integration with the rest of the code.
    if args.grayscale:
        img = colormodel.rgb2grayscale(img)

        # Add a fake 'channel' dimension. This makes it easier to make grayscale images interact with the rest of the code.
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, repeats=3, axis=2)


    # If the user wants to quantize the image. args.quantize contains the number of colors available.
    if args.quantize is not None:
        # Creates a uniformily spaced color distribution. It's a uniform division from 0 to 255, with args.quantize different colors.
        availableColors = np.linspace(0, 255, args.quantize, dtype=np.uint8)
        
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
            hsvImg   = colormodel.rgb2hsv(img)

            # This is kinda crazy, but we have to use separate functions depending if the image is Grayscale or if it is RGB.
            # That's because if the image is in grayscale, then the available colors are... well... the array availableColors.

            # But if the image is RGB, then the available colors are all the unique values in the Hue channel.
            # That's because even though we quantize the image with an arbitrary number of colors, that reduced number of
            # colors can COMBINE INTO DIFFERENT colors because of the 3 channels. For example, if there's only 3 colors for each channel:
            # [0, 127, 255], then there's 3 * 3 * 3 different combinations of colors.
            # This is what ends up giving us a very large number of different Hues, and the reason why
            # the colors available in the RGB image are the unique values in hsvImg[..., 0] instead of availableColors :)
            if args.grayscale:
                colorLUT = colormapping.generatePalette(args.hue, availableColors, args.hue_range, args.hue_reversed)
                hsvImg   = colormapping.changeColorPaletteGrayscale(hsvImg, colorLUT)
            else:
                colorLUT = colormapping.generatePalette(args.hue, np.unique(hsvImg[..., 0]), args.hue_range, args.hue_reversed)
                hsvImg   = colormapping.changeColorPaletteRGB(hsvImg, colorLUT)
                

            img  = colormodel.hsv2rgb(hsvImg)

    if args.convolution is not None:
        kernerlMap = {
                        "boxblur3x3": kernels.boxBlur3x3,
                        "boxblur5x5": kernels.boxBlur5x5,
                        "gaussian3x3": kernels.gaussianBlur3x3,
                        "gaussian5x5": kernels.gaussianBlur5x5
        }

        kernel = kernerlMap[args.convolution]
        img    = np.stack([convolve2d.convolve2d(img[..., channel], kernel) for channel in range(img.shape[-1])], axis=2)


    # Save the image
    img = PIL.Image.fromarray(img)
    img.save("./processed.png")


if __name__ == '__main__':
    args = parser.make_parser().parse_args()
    
    main(args)