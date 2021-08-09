import glob
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.utils import save_image

from models.vae.msp import MSP
from utils import get_parser_model_flow

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
save_path = args.flow_path / Path('media') / Path('feature-manipulation')
save_path.mkdir(exist_ok=True)

flow = torch.load(args.flow_path / Path('checkpoints/model.pkt')).to(DEVICE).eval()
model = MSP(256, 40, 64, nc=3)
model.load_state_dict(torch.load(args.model_path / Path('checkpoints/MSP_CelebA.tch'), map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset_valid = CelebA(root='data', split="valid", transform=transform, download=False)
dataloader_valid = DataLoader(dataset_valid, batch_size=10)

with torch.no_grad():
    x, y = next(iter(dataloader_valid))
    x, y = x.to(DEVICE), y.to(DEVICE)
    mu, log_var = model.encode(x)
    z = model.reparameterize(mu, log_var)

    idx = 1

    image = (x[idx] + 1) / 2
    image_recon = (model.decoder(z[idx:idx + 1]) + 1) / 2
    save_image(image, save_path / Path('image.png'), nrow=1, padding=0)
    save_image(image_recon, save_path / Path('image_recon.png'), nrow=1, padding=0)

    y_image_org = y[idx:idx + 1].clone().float()
    noise = flow.transform_to_noise(z[idx:idx + 1], y_image_org)

    for idx, label in enumerate(CLASSES):
        print(f"Processing feature number {idx} - {label}")
        y_image_chg = y_image_org.clone().detach()
        print(y_image_chg)
        new_value = 1.0 if y_image_chg[0][idx] == 0.0 else 0.0
        y_image_chg[0][idx] = new_value

        embedded_context = flow._embedding_net(y_image_chg)
        samples, _ = flow._transform.inverse(noise, context=embedded_context)

        image_chg = model.decoder(samples)
        image_chg = (image_chg[0] + 1) / 2
        save_image(
            image_chg,
            save_path / Path(f"{idx}_{label}_{new_value}.png"),
            nrow=1,
            padding=0
        )

for file in glob.glob(f'{save_path}/*.png'):
    filename = file.split('/')[-1]
    command = ["montage", "-mode", "concatenate", f"{save_path}/image.png", f"{save_path}/image_recon.png", file,
               f"{save_path}/Mosaic-{filename}.png"]
    print(command)
    subprocess.run(command)
