def add_image_options(options):
    # Add image params to option parser
    options.parser.add_argument('--normalize', type=int, default=1,
                                help='use imageNet normalization values\
                                for input during training')
