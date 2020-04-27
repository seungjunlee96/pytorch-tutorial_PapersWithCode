# Neural Style Transfer

[Neural style transfer](https://arxiv.org/abs/1508.06576) is an algorithm that combines the content of one image with the style of another image using CNN. Given a content image and a style image, the goal is to generate a target image that minimizes the content difference with the content image and the style difference with the style image. 

<p align="center"><img width="100%" src="png/neural_style2.png" /></p>


# The principles of Neural Style Transfer
- **Processing hierarchy of the network** : Intermediate layers represent feature maps that become increasingly higher ordered as you go deeper.
- **Content and Style representations** : the representations of content and style in the CNN are separable
- Calculate the feature distance between content image and style image by features extracted from intermediate layers of VGG19 which is pretrained on ImageNet.
- high level content : Higher layers of CNN lose the detailed pixel information while the high-level content of the image is preserved.

```python
class VGGNet(torch.nn.Module):
    def __init__(self):
        """select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet ,self).__init__()
        self.feature_select = ['0' , '5' , '10' , '19' , '28']
        self.vgg = torchvision.models.vgg19(pretrained = True).features

def forward(self ,x):
    """Extract multiple convolutional feature maps."""
    features = []
    for name, layer in self.vgg._modules.items():
        if name in self.features_select:
            feature = layer(x)
            features.append(feature)
    return features
```

#### Neural Style Transfer images:
- A content image (c) : the image we want to transfer a style to
- A style image (s) : the image we want to transfer a style from
- An generated(target) image : the image that contains the final result (**the only trainable variable**)



#### Content loss : Defined by MSE(Mean Squared Error) between content image feature and generated image feature from intermediate layers of VGG19.
To minimize the content difference, we forward propagate the content image and the target image to pretrained [VGGNet](https://arxiv.org/abs/1409.1556) respectively, and extract feature maps from multiple convolutional layers. Then, the target image is updated to minimize the [mean-squared error](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/neural_style_transfer/main.py#L81-L82) between the feature maps of the content image and its feature maps. 

```python
target_features = vgg(target)
content_features = vgg(content)

content_loss = torch.mean((target_features - content_features)**2)
```

#### Style loss
- By minimising the MSE between the entries of the Gram matrix from the original image and the Gram matrix of the image to be generated.

As in computing the content loss, we forward propagate the style image and the target image to the VGGNet and extract convolutional feature maps. To generate a texture that matches the style of the style image, we update the target image by minimizing the mean-squared error between the Gram matrix of the style image and the Gram matrix of the target image (feature correlation minimization). See [here](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/neural_style_transfer/main.py#L84-L94) for how to compute the style loss.

```python
target_features = vgg(target)
style_features = vgg(style)

style_loss = 0.0
for target_feature , style_feature in zip(target_features, style_features):
  target_gram_feature = torch.mm(target_feature, target_feature.t())
  style_gram_feature = torch.mm(style_feature, style_feature.t())
  style_loss += torch.mean((target_gram_feature - style_gram_feature)**2) / (c * h * w)
```
#### Total Loss
As a result, the total loss is given as the weighted sum of content loss and style loss.
```python
loss = content_loss + style_weight * style_loss
```
#### Details
- Max pooling -> Average Pooling : Replacing the **Max Pooling** operation by **Average Pooling** improves the gradient flow and obtains slightly more appealing results.

<br>

## Usage 

```bash
$ pip install -r requirements.txt
$ python main.py --content='png/content.png' --style='png/style.png'
```

<br>

## Results
The following is the result of applying variaous styles of artwork to Anne Hathaway's photograph.

![alt text](png/neural_style.png)
