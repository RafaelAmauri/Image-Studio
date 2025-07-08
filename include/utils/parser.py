from argparse import ArgumentParser


def make_parser():
    parser     = ArgumentParser(description="Define the parameters")

    parser.add_argument('-i', '--image', type=str, required=True,
                        help="The image that is going to be processed.")

    parser.add_argument('-q', '--quantize', type=int, default=255,
                        help='Quantizes the image according to an arbitrary number of colors. Does NOT dither the image, so expect major color banding.')

    parser.add_argument('-d', '--dithering', choices=["ordered", "floyd-steinberg"], default=None,
                        help='Quantizes the image, but this time applying dithering to the image to help minimize color banding. \
                            The choices are either ordered dithering or floyd-steinberg dithering.')

    parser.add_argument('-g', '--grayscale', action='store_true', default=False,
                        help='Converts the image to grayscale before processing. The output will also be a grayscale image.')

    parser.add_argument('--hue', type=int, default=None,
                        help="Specify a hue value (google HSV color wheel) to convert the image to a different color palette. \
                            Super recommended to also use the -g option, because converting the color palette of an RGB image \
                                tends to give weird results.")

    parser.add_argument('--hue-range', type=int, default=0,
                       help="By how much the hue in the color palette can vary. Default = 0.")
    
    parser.add_argument('--hue-reversed', action='store_true', default=False,
                        help="Reverses the color pallete. Instead of [hue - hue_range, hue + hue_range], it changes to [hue + hue_range, hue - hue_range].")

    parser.add_argument('--blur', '-b', type=str, choices=["boxblur3x3", "boxblur5x5", "gaussian3x3", "gaussian5x5"], default=None,
                        help="Apply a blur filter in the image. Choose from the available implemented blur kernels.")

    parser.add_argument('--edge-detection', '-e', type=str, choices=["sobel", "prewitt"], default=None, 
                        help="Detects edges in the image using one of the available algorithms.")

    parser.add_argument('--edge-color', '-ec', type=int, default=-1, 
                        help="Colors the detected edges with a specific color. -2 = All edges are white, -1 = Assigns a Hue value based on \
                            the direction that the edges points to, any other value = colors all edges with that Hue value. Default = -1")

    parser.add_argument('--contrast', '-c', type=float, default=-1,
                        help="By how much to boost the contrast in the image. -c = 2 will take the 2% lowest and 2% highest colors" \
                        "and equal them to 0 and 255 respectively and then scale the midtones.")

    return parser


def validateParams(args):
    if args.edge_color < -2 or args.edge_color > 360:
        raise ValueError("--edge-color must be between -2 and 360")
    
    if args.hue_range < 0 or args.hue_range > 179:
        raise ValueError("--hue-range must be between 0 and 170")