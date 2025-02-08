from argparse import ArgumentParser


def make_parser():
    parser     = ArgumentParser(description="Define the parameters")

    parser.add_argument('-i', '--image', type=str, required=True,
                        help="The image that is going to be processed.")

    parser.add_argument('-q', '--quantize', type=int, default=255,
                        help='Quantizes the image according to an arbitrary number of colors. Default is 255')

    parser.add_argument('-d', '--dithering', action='store_true', default=False,
                        help='Applies dithering to the image to help mitigate a narrow color palette.')

    parser.add_argument('-g', '--grayscale', action='store_true', default=False,
                        help='Converts the image to grayscale before processing. The output will also be a grayscale image.')

    return parser