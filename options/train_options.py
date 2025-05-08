from argparse import ArgumentParser

class InferOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.register_options()

    def register_options(self):
        self.parser.add_argument('--config', type=str, default='configs/main.yaml', help='Path to the config file.')
        self.parser.add_argument('--output_path', type=str, default='.', help="outputs path")
        self.parser.add_argument("--resume", action="store_true")
        self.parser.add_argument("--gpus", nargs='+')

    def parse_args(self):
        opts = self.parser.parse_args()
        return opts