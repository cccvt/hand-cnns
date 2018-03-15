def add_stack_options(options):
    parser = options.parser
    # Add video params to option parser
    parser.add_argument('--stack_nb', type=int,
                        default=10, help='How many frames to stack')
