import argparse


def get_parser_experiment():
    parser = argparse.ArgumentParser(description='Generic runner for Flow models')
    parser.add_argument(
        '--config', '-c',
        dest="filename",
        metavar='FILE',
        help='path to the config file',
        default='configs/vae.yaml'
    )
    return parser


def get_parser_latent_space():
    parser = argparse.ArgumentParser(description='Generic runner for Flow models')
    parser.add_argument(
        '--path', '-p',
        dest="path",
        metavar='FILE',
        help='path to the model directory in logs'
    )
    return parser


def get_parser_flow_visualization():
    parser = argparse.ArgumentParser(
        description='Visualization of the conditional generation of the flow for MNIST Experiment.'
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
