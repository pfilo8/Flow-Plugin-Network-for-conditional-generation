<h1 align="center">
  <b>PyTorch VAE</b><br>
</h1>

VAE based on [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE/)
Flow based on [nflows](https://github.com/bayesiains/nflows)

### Classes

- 0 5_o_Clock_Shadow
- 1 Arched_Eyebrows
- 2 Attractive
- 3 Bags_Under_Eyes
- 4 Bald
- 5 Bangs
- 6 Big_Lips
- 7 Big_Nose
- 8 Black_Hair
- 9 Blond_Hair
- 10 Blurry
- 11 Brown_Hair
- 12 Bushy_Eyebrows
- 13 Chubby
- 14 Double_Chin
- 15 Eyeglasses
- 16 Goatee
- 17 Gray_Hair
- 18 Heavy_Makeup
- 19 High_Cheekbones
- 20 Male
- 21 Mouth_Slightly_Open
- 22 Mustache
- 23 Narrow_Eyes
- 24 No_Beard
- 25 Oval_Face
- 26 Pale_Skin
- 27 Pointy_Nose
- 28 Receding_Hairline
- 29 Rosy_Cheeks
- 30 Sideburns
- 31 Smiling
- 32 Straight_Hair
- 33 Wavy_Hair
- 34 Wearing_Earrings
- 35 Wearing_Hat
- 36 Wearing_Lipstick
- 37 Wearing_Necklace
- 38 Wearing_Necktie
- 39 Young

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

Download PointFlow scripts
```bash
git clone git@github.com:stevenygd/PointFlow.git
```
