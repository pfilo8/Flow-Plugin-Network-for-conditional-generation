# Plug-in networks for generatvie models

VAE based on [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE/)
Normalizing Flows based on [nflows](https://github.com/bayesiains/nflows)
PointFlow based on [PointFlow](https://github.com/stevenygd/PointFlow)
Point Cloud Renderer based on [Point Flow Renderer](https://github.com/zekunhao1995/PointFlowRenderer)

### Usage
```
$ cd PyTorch-VAE
$ python run.py -c configs/<config-file-name.yaml>
```
**Config file template**
```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: 3
  latent_dim: 
    .         # Other parameters required by the model
    .
    .

exp_params:
  data_path: "<path to the celebA dataset>"
  img_size: 64    # Models are designed to work for this size
  batch_size: 64  # Better to have a square number
  LR: 0.005
  weight_decay:
    .         # Other arguments required for training, like scheduler etc.
    .
    .

trainer_params:
  gpus: 1         
  max_nb_epochs: 50
  gradient_clip_val: 1.5
    .
    .
    .

logging_params:
  save_dir: "logs/"
  name: "<experiment name>"
  manual_seed: 
```

**View TensorBoard Logs**
```
$ cd logs/<experiment name>/version_<the version you want>
$ tensorboard --logdir tf
```

