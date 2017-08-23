from src.options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # Weight loading
        self.parser.add_argument('--checkpoint_path', type=str, default='',
                                 help='location of checkpoint to load')
