from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import get_parser_model_flow

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = get_parser_model_flow()
parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
args = parser.parse_args()
save_path = args.flow_path / Path('samples')
save_path.mkdir(exist_ok=True)

flow = torch.load(args.flow_path / Path('checkpoints/model.pkt')).to(DEVICE).eval()

n_samples = args.n_samples
num_classes = 55 if 'shapenet' in args.flow_path else 10
n_row = 8

outputs_samples = []
outputs_ys = []

for i in range(num_classes):
    context = torch.zeros(1, num_classes).to(DEVICE)
    context[0][i] = 1.0
    print(context)

    with torch.no_grad():
        samples = flow.sample(n_samples, context).squeeze(0).detach().numpy()
        outputs_samples.append(samples)
        outputs_ys.append(int(i) * np.ones((samples.shape[0], 1), dtype=np.int8))

df_samples = pd.DataFrame(np.vstack(outputs_samples))
df_ys = pd.DataFrame(np.vstack(outputs_ys))

df_samples.to_csv(save_path / Path('xs.csv'), index=False)
df_ys.to_csv(save_path / Path('ys.csv'), index=False)
