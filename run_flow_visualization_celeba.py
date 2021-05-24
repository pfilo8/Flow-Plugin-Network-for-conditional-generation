from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

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


def show_image(image, filepath):
    plt.imshow(image.squeeze(0).permute(1, 2, 0).detach().cpu())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)


args = get_parser_model_flow().parse_args()
save_path = args.flow_path / Path('media')

flow = torch.load(args.flow_path / Path('checkpoints/model.pkt')).to(DEVICE).eval()
model = MSP(256, 40, 64, nc=3)
model.load_state_dict(torch.load(args.model_path / Path('checkpoints/MSP_CelebA.tch'), map_location=DEVICE))
model.to(DEVICE)
model.eval()

n_samples = 16
n_row = 8

outputs = []

with torch.no_grad():
    for idx, label in enumerate(CLASSES):
        context = torch.zeros(1, 40).to(DEVICE)
        context[0][idx] = 1.0
        print(context)

        samples = flow.sample(n_samples, context).squeeze(0)
        output = model.decoder(samples)
        output = output.add_(1.0).div_(2.0)
        outputs.append(output)
        # outputs.append(output)
        show_image(
            (make_grid(output, nrow=n_row, padding=0)),
            filepath=save_path / Path(f"{label}.png")
        )

    show_image(
        make_grid(torch.cat(outputs)[::4], 16),
        filepath=save_path / Path("all-16.png")
    )

    show_image(
        make_grid(torch.cat(outputs)[::2], 32),
        filepath=save_path / Path("all-32.png")
    )
