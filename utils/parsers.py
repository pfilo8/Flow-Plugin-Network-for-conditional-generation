import argparse


def get_parser_experiment():
    parser = argparse.ArgumentParser(description='Generic runner for Flow models')
    parser.add_argument(
        '--config', '-c',
        dest="filename",
        metavar='FILE',
        help='Path to the config file'
    )
    return parser


def get_parser_model_flow():
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--model_path',
        metavar='FILE',
        help='Model path'
    )
    parser.add_argument(
        '--flow_path',
        metavar='FILE',
        help='Flow path'
    )
    return parser
