import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PointFlow.args import get_parser
from PointFlow.models.networks import PointFlow

from utils import get_parser_model_flow
from utils import translate_point_cloud_to_xml


def load_model(model_dir):
    parser = get_parser()
    args = parser.parse_args([
        '--cates',
        'all',
        '--resume_checkpoint',
        f'{model_dir}/checkpoints/model.pt',
        '--dims',
        '512-512-512',
        '--use_deterministic_encoder',
        '--evaluate_recon',
        '--resume_dataset_mean',
        f'{model_dir}/checkpoints/train_set_mean.npy',
        '--resume_dataset_std',
        f'{model_dir}/checkpoints/train_set_std.npy'
    ])

    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model.multi_gpu_wrapper(_transform_)

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    _ = model.eval()
    return model


def load_flow(path):
    flow = torch.load(path, map_location=torch.device('cpu'))
    return flow


def load_data(args):
    save_dir = os.path.dirname(args.resume_checkpoint)
    path_df_train = os.path.join(save_dir, 'train_latent_space.csv')
    path_df_test = os.path.join(save_dir, 'test_latent_space.csv')

    df_train = pd.read_csv(path_df_train)
    df_test = pd.read_csv(path_df_test)
    return df_train, df_test


args = get_parser_model_flow().parse_args()

flow_path = Path(args.flow_path)
save_path = flow_path / Path('media')
save_path.mkdir(exist_ok=True)

model = load_model(args.model_path)
flow = load_flow(flow_path / Path('checkpoints/model.pkt'))

N = 2048 * 4
NUM_CLASSES = 55

classes = ['airplane', 'bag', 'basket', 'bathtub', 'bed', 'bench',
           'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet',
           'camera', 'can', 'cap', 'car', 'cellphone', 'chair', 'clock',
           'dishwasher', 'earphone', 'faucet', 'file', 'guitar', 'helmet',
           'jar', 'keyboard', 'knife', 'lamp', 'laptop', 'mailbox',
           'microphone', 'microwave', 'monitor', 'motorcycle', 'mug', 'piano',
           'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle',
           'rocket', 'skateboard', 'sofa', 'speaker', 'stove', 'table',
           'telephone', 'tin_can', 'tower', 'train', 'vessel', 'washer']

for label, cls in enumerate(classes):
    print(f'[{label + 1} / {len(classes)}] Generating {cls}.')
    context = torch.zeros((1, NUM_CLASSES))
    context[0, label] = 1

    sample = flow.sample(1, context).squeeze(0)
    result = model.decode(z=sample, num_points=N)

    generated_samples = result[1].cpu().detach().numpy()
    translate_point_cloud_to_xml(generated_samples[0], save_path / Path(f'{cls}.xml'))
