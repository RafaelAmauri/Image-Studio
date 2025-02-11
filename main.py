import numpy as np
import PIL.Image

import include.colormapping as colormapping
import include.colorspace as colorspace
import include.quantize as quantize
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

    # Quantize the image (with or without dithering)
    if args.quantize != -1:
        # Creates a uniformily spaced color distribution. It's a uniform division from 0 to 255, with args.quantize different colors.
        availableColors = np.linspace(0, 255, args.quantize, dtype=np.uint8)
        img = quantize.quantize(img, availableColors, useDithering=args.dithering)
    
    # Convert the color palette according to a color LUT
    colorLUT = { np.float32(0/255) :  [93, 29/100, 12/100],
                 np.float32(85/255):  [124, 52/100, 25/100],
                 np.float32(170/255): [114, 45/100, 35/100],
                 np.float32(255/255): [149, 37/100, 93/100]
                 }
    
    hsvImg = colorspace.rgb2hsv(img)
    hsvImg = colormapping.changePalette(hsvImg, colorLUT)
    img = colorspace.hsv2rgb(hsvImg)

    # Save the image
    img = PIL.Image.fromarray(img)
    img.save("./processed.png")


if __name__ == '__main__':
    args = parser.make_parser().parse_args()
    
    main(args)