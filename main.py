import numpy as np
import PIL.Image

import include.quantize as quantize
import include.parser as parser


def main(args):
    enableGrayscale = args.grayscale

    # Open the image
    img = PIL.Image.open(args.image)

    # Convert to grayscale if so desired
    if enableGrayscale:
        img = img.convert("L")

    img = np.asarray(img, dtype=np.uint8)

    # To make the code work with grayscale, we need to add a fake "channel" dimension. This makes 
    # it easier to interact with the rest of the code
    if enableGrayscale:
        img = np.expand_dims(img, axis=0)

    # Convert from [H, W, C] into the [C, H, W] format
    img = img.transpose(2, 0, 1)

    # Quantize the image (with or without dithering)
    img = quantize.quantize(img, numberOfColors=args.quantize, useDithering=args.dithering)


    # Convert from [C, H, W] back into [H, W, C]
    img = img.transpose(1, 2, 0)

    # Remove the fake "channel" dimension
    if enableGrayscale:
        img = img.squeeze(0)


    # Save the image
    img = PIL.Image.fromarray(img)
    img.save("./processed.png")


if __name__ == '__main__':
    args = parser.make_parser().parse_args()
    
    main(args)