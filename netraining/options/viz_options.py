def add_viz_options(options):
    """
    Add options for neural network introspection
    """
    parser = options.parser
    parser.add_argument(
        '--level', type=str, default='', help='for i3d in [2a|3a|4a|5a]')
    parser.add_argument(
        '--activation_idx',
        nargs='+',
        default=[],
        help='List of activation idxs')
