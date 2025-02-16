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
            # TODO Write why
            if args.grayscale:
                colorLUT = colormapping.generatePalette(args.hue, availableColors, args.hue_range, args.hue_reversed)
                hsvImg   = colormapping.changeColorPaletteGrayscale(hsvImg, colorLUT)
            else:
                # Convert the hue to the range [0, 65535]
                hueChannelUint16 = (hsvImg[..., 0] / 360 * 65535).astype(np.uint16)
                colorLUT = colormapping.generatePalette(args.hue, np.unique(hueChannelUint16), args.hue_range, args.hue_reversed)
                hsvImg   = colormapping.changeColorPaletteRGB(hsvImg, colorLUT, hueChannelUint16)
                

            img  = colormodel.hsv2rgb(hsvImg)


    # Save the image
    img = PIL.Image.fromarray(img)
    img.save("./processed.png")


if __name__ == '__main__':
    args = parser.make_parser().parse_args()
    
    main(args)