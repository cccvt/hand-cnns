def add_test_options(options):
    parser = options.parser
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        help='location of checkpoint to load')
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='location of checkpoint to load')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help="Split to use at testing time for\
                        smthg-smthg dataset")
    parser.add_argument(
        '--mode',
        type=str,
        default='stride',
        help="Mode for testing in [full|stride|subsample]\
        full for fully convolutional, stride for fixed stride offset\
        subsample for uniform sampling in clip")
    parser.add_argument(
        '--mode_param',
        type=int,
        default='8',
        help="Time stride when mode is stride, number of samples\
        when mode is subsample")
    parser.add_argument('--epoch', type=str, help='Index of epoch to load')
