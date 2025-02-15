from argparse import ArgumentParser


def make_parser():
    parser     = ArgumentParser(description="Define the parameters")

    parser.add_argument('-i', '--image', type=str, required=True,
                        help="The image that is going to be processed.")

    parser.add_argument('-q', '--quantize', type=int, default=None,
                        help='Quantizes the image according to an arbitrary number of colors. Does NOT dither the image, so expect major color banding.')

    parser.add_argument('-d', '--dithering', choices=["ordered", "floyd-steinberg"], default=None,
                        help='Quantizes the image, but this time applying dithering to the image to help minimize color banding. The choices are either ordered dithering or floyd-steinberg dithering.')

    parser.add_argument('-g', '--grayscale', action='store_true', default=False,
                        help='Converts the image to grayscale before processing. The output will also be a grayscale image.')

    parser.add_argument('-p', '--palette', type=int, default=None,
                        help="Specify a hue value (google HSV color wheel) to convert the image to a different color palette. Super recommended to also use the -g option, because converting the color palette of an RGB image tends to give weird results.")

    parser.add_argument('-pr', '--palette-range', type=int, default=0,
                       help="By how much the color palette can vary. Default = 0")
    

    return parser