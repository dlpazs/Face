# Deformable Convolutions
- We can deform or occlude objects that we want to detect or classify within an image
- It works by a normal convolution offset by some value
- so a branch of the normal convolution is used to offset the original convolution
![alt text](https://miro.medium.com/max/1373/1*Mi6LqBIa8a4Ewo9DywHuzw.png)
- the deformable convolution picks values at different locations to do convolutions conditioned on the input (input image or feature map)
-

[Different types of convolutions](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)
- Atrous (dilated convolutions) : a space between values in a kernel which creates a wider field of view

- Transposed (fractionally strided convolutions) : this is like inverting a convolution after you've done a convolution but not
equivalent in inverting the maths required to do the initial convolution. Say we did
a convolution which produced a 2x2 image then how would we go back to the original image from that? Well all it does is guarantee a certain
size of an output image by lots of padding. It reconstructs the spatial resolution from before and performs a convolution. Useful for
encoder/decoder architectures (a bit like an autoencoder whereby we reduce the input to a lower dimensional manifold and try and
re-construct the original image).


- Depthwise separable convolution: Say we have a 3x3 conv layer on 16 input channels and 32 output channels. Every channel
of the 16 is traversed by 32 3x3 kernels resulting in 512 (16x32) feature maps. Then we merge 1 feature map out of every
input input channel by adding them up. Since we do that 32 times we get 32 output channels. For a depthwise separable
convolution, we traverse the 16 channels with 1 3x3 kernel each giving us 16 feature maps. Before merging anything, we
traverse these 16 feature maps with 32 1x1 convolutions each and only then add them together. Giving 656 (16x3x3 +
16x32x1x1) parameters opposed to 4608 (16x32x3x3). 

[Basic into to seperable convs](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- Spatial Separable Convolutions :
the basic idea is separating one convolution into two. It deals with the spatial dimension (h, w). It divides a kernel
into two smaller kernels. So a 3x3 kernel into a 3x1 kernel and a 1x3 kernel. 
![alt text](https://miro.medium.com/max/1050/1*o3mKhG3nHS-1dWa_plCeFw.png)
- Now instead of a conv with 9 multiplications, we do two convolutions with 3 multiplications each (6 total). Reducing
computational complexity.
- One famous conv that can be separated spatially is the Sobel kernel used to detect edges : 
![alt text](https://miro.medium.com/max/1050/1*r4MjVvb2rehlIzpTKZdhcA.png)
- The problem is that not all kernels can be separated into two smaller kernels. 
- Depthwise separable convs: These can work with kernels that cannot be factored into smaller kernels. In keras, 
`keras.layers.SeparableConv2D` or `tf.layers.separable_conv2d`
- It deals with spatial and depth dimension
- An input image may have 3 channels RGB but after a few convs many channels
- A channel is an interpretation of that image
- It splits a kernel to 2 separate kernels that do 2 convs: the depthwise convolution and pointwise convolution
- Normal Convolution: Say an image is 12x12x3. A 5x5 conv with no padding stride 1. This results in 8x8 output since
12 - 5 + 1 = 8. Then 3 channels we do 5x5x3 = 75. In 2-D we'd get 8x8x1 from a 5x5 kernel on 12x12x3. 
What if we want to increase the number of output channels to 256. We just create 256 kernels to create 8x8x256.
It is not a matrix multiplication but sliding dot product elementwise.
- Depthwise convolution: From the same example: a 5x5x1 kernel iterates 1 channel of the image
![alt text](https://miro.medium.com/max/1050/1*yG6z6ESzsRW-9q5F_neOsg.png)
- getting the scalar products of every pixel group outputting a 8x8x1 image stacking them 8x8x3
- Pointwise Convolution: So the depthwise conv has transformed the 12x12x3 to 8x8x3, now we need to increase the number
of channels 
- the pointwise conv is named because it uses a 1x1 kernel, a kernel that iterates through every single point and
it has a depth of however many channels of its input so 3. 
![alt text](https://miro.medium.com/max/1050/1*37sVdBZZ9VK50pcAklh8AQ.png)
- so we iterate through 8x8x3 with 1x1x3 to get 8x8x1. If we create 256 of these 1x1x3 kernels we get 8x8x256
- now we've separated the conv into a depthwise and pointwise conv
- Whats the point? We've reduced the amount of multiplications from 1.2m to 53,000.
- the intuition is in a normal conv we transform the image 256 times, and in separable we only transform once in
the depthwise conv then take the transformed and elongate it to 256 channels.
- In keras and tf there is an arg called depth multiplier, 1 by default. By changing we can set the number of output
channels in depthwise conv. If we set to 2 each 5x5x1 kernel will output 8x8x2

