# Deep Residual Learning for Image Recognition : [[Link](https://arxiv.org/pdf/1512.03385.pdf)]
Hierachical Features and Role of Depth
- Low, mid, and high-level features
- more layers enrich the **"levels"** of the features
- Previous ImageNet models have depths of 16 and 30 layers

Previous Approaches : *"Is learning better networks as easy as stacking more layers?"*
- Vanishing/Exploding gradients : Authors argue that this is **unlikely** to cause the problem of optimization difficulty where BN ensures healty norms.
- Network *degradation* : With the network depth increasing, accuracy gets saturated and then degrades rapidly. (but it is *not caused by overfitting.*)
- Adding more layers to fa suitably deep model leads to a *higher training error*.

The degradation problem suggests that the sovlers might have **difficulties in approximating identity mappings** by multiple nonlineaer layers.

As a solution to the above problems, this paper introduced a ***"deep residual learning"*** framework.
- **Residual Mapping: H(x) = F(x) + x** ( Conv block + Identity block)
- Conv block : Conv > BN > ReLU > Conv > BN > ReLU > Conv > BN
- 152 layers ( 8 x times deeper than VGG19)
- By adding **identity layers**, deep networks should perform at least as well as a smaller network. (In other words, identity layers solve the degradation problem.)


ResNet : Two kinds of Residual Blocks
1. H(x) = F(x) + x
2. H(x) = F(x) + WX (if the input shape changes after , operation W (Conv > BN) matches the shape)


Overall Network design
1. for the same output feature map size, the layers have the same number of filters
2. if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer.
3. perform downsampling directly by convolutional layers that have a stride of 2.(Not MaxPooling or AvgPooling)
4. ends with a global average pooling layer and a 1000-way fully-connected layer with softmax
