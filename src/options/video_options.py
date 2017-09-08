def add_video_options(options):
    parser = options.parser
    # Add video params to option parser
    parser.add_argument('--clip_spacing', type=int,
                        default=1, help='When using clip, how many frames to skip\
                        between consecutive frames')
