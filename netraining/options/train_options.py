def add_train_options(options):
    parser = options.parser
    parser.add_argument(
        '--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument(
        '--weighted_training',
        action='store_true',
        help="Use weighted sampling during training")

    # Valid params
    parser.add_argument(
        '--test_aggreg',
        type=int,
        default=1,
        help="0 to disable aggregation test during\
                             training")

    # Net params
    parser.add_argument(
        '--pretrained',
        type=int,
        default=1,
        help="Use pretrained weights for net (1)")

    # Optim params
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Base learning rate for training')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Base learning rate for training')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay for training')
    parser.add_argument(
        '--criterion',
        type=str,
        default='CE',
        help='(MSE for mean square |\
                             CE for cross-entropy)')
    parser.add_argument(
        '--plateau_scheduler',
        action='store_true',
        help='Use learning rate reduce on plateau learning '
        'rate scheduler')
    parser.add_argument(
        '--plateau_thresh',
        type=float,
        default=1e-4,
        help='Value for relative reductin of lr in plateau')
    parser.add_argument(
        '--plateau_factor',
        type=float,
        default=0.5,
        help='Factor for lr reduction in plateau')
    parser.add_argument(
        '--plateau_patience',
        type=int,
        default=2,
        help='Number of epochs with bad metric before drop '
        'in lr')

    parser.add_argument(
        '--save-freq',
        type=int,
        default=5,
        help='Frequency at which to save the \
                             network weights')
    parser.add_argument(
        '--save-latest',
        type=int,
        default=1,
        help='Whether to save the latest computed weights \
                             at each epoch')

    # Load params
    parser.add_argument(
        '--continue_training',
        action='store_true',
        help='Continue training from saved weights')
    parser.add_argument(
        '--continue_epoch',
        type=int,
        default=0,
        help='Epoch to load for trianing continuation \
                             latest if 0')

    # Display params
    parser.add_argument(
        '--display_freq',
        type=int,
        default=100,
        help='number of iters between displays of results\
                             in visdom')

    # Multi training params
    parser.add_argument(
        '--multi_weights',
        nargs='+',
        type=float,
        help='Weights for each loss term in mutlitraining')
    parser.add_argument(
        '--multi_prefixes',
        nargs='+',
        type=str,
        help='Prefixes for each loss term in mutlitraining')

    parser.add_argument(
        '--mini_factor', type=float, help='Work with small version of dataset')
