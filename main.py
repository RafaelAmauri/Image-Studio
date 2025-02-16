import numpy as np
import PIL.Image

import include.colormapping as colormapping
import include.colormodel as colormodel
import include.quantize as quantize
import include.dither as dither
import include.parser as parser


def main(args):

    # Open the image
    img = PIL.Image.open(args.image)

    # Convert to grayscale if so desired. The change back to RGB is to add a 3-channel dimension to the image.
    # This simplifies the integration with the rest of the code.
    if args.grayscale:
        img = img.convert("L")
        img = img.convert("RGB")

    img = np.asarray(img, dtype=np.uint8)
    

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
            # colors can COMBINE into different colors because of the 3 channels. For example, if there's only 3 colors for each channel:
            # [0, 127, 255], then there's 3 * 3 * 3 different combinations of colors. ]
            # R = 0, G = 0, B = 0
            # R = 0, G = 0, B = 127
            # R = 0, G = 0, B = 255
            # R = 0, G = 127, B = 0
            # and so on... Which doesn't happen when the image is grayscale. So this ends up giving us a large number of different Hues.
            # And that's why the colors available are all the unique values in hsvImg[..., 0] :)
            
            if args.grayscale:
                colorLUT = colormapping.generatePalette(args.hue, availableColors, args.hue_range, args.hue_reversed)
                hsvImg   = colormapping.changeColorPaletteGrayscale(hsvImg, colorLUT)
            else:
                colorLUT = colormapping.generatePalette(args.hue, np.unique(hsvImg[..., 0]), args.hue_range, args.hue_reversed)
                hsvImg   = colormapping.changeColorPaletteRGB(hsvImg, colorLUT)
                

            img  = colormodel.hsv2rgb(hsvImg)


    # Save the image
    img = PIL.Image.fromarray(img)
    img.save("./processed.png")


if __name__ == '__main__':
    args = parser.make_parser().parse_args()
    
    main(args)