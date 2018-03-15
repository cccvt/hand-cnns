def add_video_options(options):
    parser = options.parser
    # Add video params to option parser
    parser.add_argument(
        '--clip_spacing',
        type=int,
        default=1,
        help='When using clip, how many frames to skip\
                        between consecutive frames')
    parser.add_argument(
        '--clip_size',
        type=int,
        default=16,
        help='When using clip, how many frames to use')
    parser.add_argument(
        '--network',
        type=str,
        default='c3d',
        help='Network to use for training [i3d|c3d|i3dense|i3res]')
