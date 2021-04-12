<h1 align="center">
  <b>PyTorch VAE</b><br>
</h1>

VAE based on [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE/)
Flow based on [nflows](https://github.com/bayesiains/nflows)

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

### Citation
```
@misc{Subramanian2020,
  author = {Subramanian, A.K},
  title = {PyTorch-VAE},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
}
```
-----------
