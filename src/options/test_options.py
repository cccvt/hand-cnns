def add_test_options(options):
    parser = options.parser
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='location of checkpoint to load')
