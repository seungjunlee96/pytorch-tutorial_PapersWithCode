# Variational Autoencoder
## Papers:
- [The original Paper of VAE (Variational Autoencoder)](https://arxiv.org/abs/1312.6114).
- [Tutorial on Variational Autoencoder](https://arxiv.org/pdf/1906.02691.pdf) provides much more detailed explanation of VAE.

## Definition
- **Variational Inference**을 오토 인코더 구조를 통해 구현한 신경망.
- **Variational Inference** : General family of methods for **approximating** complicated densities by a simpler class of densities.
- **Variational Inference** : q(z|x) ~= p(z|x) ,where p is much simpler than q.

## Brief Summary
![VAE](./images/VAE.png)

## Autoencoder의 구조는 동일
- Encoder : Convolutional Network map given high dimensional image to **latent space representation**
- Decoder : Deconvolutional Network decompress the **latent space representation** to high dimensional image.

VAE : Autoencoder에서 더 나아가, Latent vector Z가 다루기 쉬운 "확률 분포"를 띄게 만들자!

The framework of *Variational Autoencoders*(VAEs) provides a principled method for jointly learning
- deep latent-variable models 
- corresponding inference models using stochastic gradient descent

## Reparameterization


## Comparison between VAE vs GAN
- Optimization
- Image Quality 
- Generalization






# Referenes
- [Tutorial on Variational Autoencoder](https://arxiv.org/pdf/1906.02691.pdf)
- [오토인코더의 모든 것](https://www.slideshare.net/NaverEngineering/ss-96581209)
- [Towards a Deeper Understanding of Variational Autoencoding Models](https://arxiv.org/pdf/1702.08658.pdf)
- 
