from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from nflows.utils import torchutils
from torchvision.utils import save_image

from models.vae.msp import MSP
from utils import get_parser_model_flow

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CLASSES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
    'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
    'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
    'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]
plt.ioff()

args = get_parser_model_flow().parse_args()
save_path = args.flow_path / Path('media')

flow = torch.load(args.flow_path / Path('checkpoints/model.pkt')).to(DEVICE).eval()
model = MSP(256, 40, 64, nc=3)
model.load_state_dict(torch.load(args.model_path / Path('checkpoints/MSP_CelebA.tch'), map_location=DEVICE))
model.to(DEVICE)
model.eval()

n_samples = 4
n_row = 4

outputs = []


def sample_from_flow(flow, num_samples, context=None, temperature=1.0):
    embedded_context = flow._embedding_net(context)
    noise = temperature * flow._distribution.sample(num_samples, context=embedded_context)

    if embedded_context is not None:
        # Merge the context dimension with sample dimension in order to apply the transform.
        noise = torchutils.merge_leading_dims(noise, num_dims=2)
        embedded_context = torchutils.repeat_rows(
            embedded_context, num_reps=num_samples
        )

    samples, _ = flow._transform.inverse(noise, context=embedded_context)

    if embedded_context is not None:
        # Split the context dimension from sample dimension.
        samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

    return samples


with torch.no_grad():
    for idx, label in enumerate(CLASSES):
        for t in np.linspace(0.7, 0.8, 11):
            context = torch.zeros(1, 40).to(DEVICE)
            context[0][idx] = 1.0
            print(context)

            samples = sample_from_flow(flow, n_samples, context, temperature=t).squeeze(0)
            output = model.decoder(samples)
            output = output.add_(1.0).div_(2.0)
            outputs.append(output)
            save_image(
                output,
                save_path / Path(f"{label}_{t}.png"),
                nrow=n_row,
                padding=0
            )

CONTEXTS = [
    (
        torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1]], dtype=torch.float),
        'Male No_Beard Young'
    ),
    (
        torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 0, 1]], dtype=torch.float),
        'Male No_Beard Wearing_Hat Young'
    ),
    (
        torch.tensor(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 1]], dtype=torch.float),
        'Male Mouth_Slightly_Open No_Beard Young'
    )
]

with torch.no_grad():
    for context, label in CONTEXTS:
        for t in np.linspace(0.7, 0.8, 11):
            print(context, t)
            context = context.to(DEVICE)

            samples = sample_from_flow(flow, n_samples, context, temperature=t).squeeze(0)
            output = model.decoder(samples)
            output = output.add_(1.0).div_(2.0)
            outputs.append(output)
            save_image(
                output,
                save_path / Path(f"0_{label}_{t}.png"),
                nrow=n_row,
                padding=0
            )
