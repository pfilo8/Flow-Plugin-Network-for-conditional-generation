![Teaser](assets/Image-teaser.png)
<h1 align="center">
  Flow Plugin Network for conditional generation 
  <br>
</h1>

<h4 align="center">Patryk Wielopolski, Michał Koperski, Maciej Zięba</h4>

> Generative models have gained many researcher's attention in the last years resulting in models such as StyleGAN for human face generation or PointFlow for 3D point cloud generation. However, by default, we cannot control its sampling process, i.e., we cannot generate a sample with a specific set of the attributes. The current approach is model retraining with additional inputs and different architecture, which requires time and computational resources. We propose a novel approach that enables to generate objects with a given set of attributes without retraining the base model. For this purpose, we utilize the normalizing flow models - Conditional Masked Autoregressive Flow and Conditional Real NVP, as a Flow Plugin Network (FPN).

## Highlights

  * Method for conditional object generation from a pre-trained autoencoder model, e.g., Vanilla Autoencoder, Variational Autoencoder.
  * Method works as a plugin network - doesn’t require base model retraining.
  * Method for non-generative autoencoder models enabling to generate objects.
  * Method successfully validated for various tasks including conditional generation, classification, attribute manipulation on MNIST, ShapeNet, CelebA.

## Results

### Conditional generation

![Conditional generation - MNIST](assets/mnist-sample-maf.png)


![Conditional generation - ShapeNet](assets/shapenet.png)

### Attribute manipulation
![Attribute manipulation](assets/celeba-feature-manipulation-0.png)

## Idea

## Authors
  * **Patryk Wielopolski** - Wrocław University of Science and Technology
  * **Michał Koperski** - Tooploox Ltd. - [Webpage](http://mkoperski.com) 
  * **Maciej Zięba** - Wrocław University of Science and Technology, Tooploox Ltd. - [Webpage](https://www.ii.pwr.edu.pl/~zieba/) 
