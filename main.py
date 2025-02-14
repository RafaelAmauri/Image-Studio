import numpy as np
import PIL.Image

import include.colormapping as colormapping
import include.colorspace as colorspace
import include.quantize as quantize
import include.dither as dither
import include.parser as parser


def main(args):
    enableGrayscale = args.grayscale

    # Open the image
    img = PIL.Image.open(args.image)

    # Convert to grayscale if so desired. The change back to RGB is to add a 3-channel dimension to the image.
    # This simplifies the integration with the rest of the code.
    if enableGrayscale:
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

        # Change the color palette
        if args.palette is not None:
            colorLUT = colormapping.generatePalette(args.palette, availableColors)

            hsvImg = colorspace.rgb2hsv(img)
            hsvImg = colormapping.changePalette(hsvImg, colorLUT)
            img    = colorspace.hsv2rgb(hsvImg)


    # Save the image
    img = PIL.Image.fromarray(img)
    img.save("./processed.png")


if __name__ == '__main__':
    args = parser.make_parser().parse_args()
    
    main(args)