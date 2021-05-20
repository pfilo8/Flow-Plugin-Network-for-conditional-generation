from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from utils import get_parser_model_flow

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_image(image, filepath):
    plt.figure(figsize=(16, 8))
    plt.imshow(image.squeeze(0).permute(1, 2, 0).detach().cpu())
    plt.axis('off')
    plt.savefig(filepath)


args = get_parser_model_flow().parse_args()
save_path = args.flow_path / Path('media')

flow = torch.load(args.flow_path / Path('checkpoints/model.pkt')).to(DEVICE).eval()
model = torch.load(args.model_path / Path('checkpoints/model.pkt')).to(DEVICE)

n_samples = 64
n_row = 8

outputs = []

for i in range(10):
    context = torch.zeros(1, 10).to(DEVICE)
    context[0][i] = 1.0
    print(context)

    samples = flow.sample(n_samples, context).squeeze(0)
    output = model.decode(samples)
    outputs.append(output)
    show_image(
        (make_grid(output, nrow=n_row)),
        filepath=save_path / Path(f"{i}.png")
    )

show_image(
    make_grid(torch.cat(outputs)[::4], 16),
    filepath=save_path / Path("all-16.png")
)

show_image(
    make_grid(torch.cat(outputs)[::2], 32),
    filepath=save_path / Path("all-32.png")
)
